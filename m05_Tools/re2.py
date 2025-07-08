"""
reconstruct_all.py
===================
Rewritten to perform **adaptive octree**‐based SDF sampling so that meshes
can be extracted with far fewer network evaluations than a dense uniform
grid.  The utils_deepsdf module is left untouched.

Algorithm overview
------------------
1.  Start from the training bounding box ``[-5, 5]^3`` (same as
    ``utils_deepsdf.get_volume_coords``).
2.  Recursively *sub‑divide* each cube (octree node) until either …
      • it reaches ``max_depth`` (default 8 → 256³ resolution), or
      • all eight corner SDF values have the *same sign* **and** are farther
        than ``band_width × half‐diagonal`` from the surface.
3.  After the octree is built, SDF values from *leaf* nodes are
    **rasterised** into a dense grid.  Cells that were never visited by the
    octree are initialised to ``+∞`` so they are ignored by Marching Cubes.
4.  ``skimage.measure.marching_cubes`` extracts the 0‑level set; vertices
    are re‑scaled back to world coordinates and stored as OBJ.

Configuration additions
-----------------------
* ``octree_max_depth`` (int, optional, default = 8)
* ``octree_band_width`` (float, optional, default = 0.5)
These can be added to your existing ``c04_reconstruct.yaml`` but sensible
defaults are used if they are missing.

"""

import os
import yaml
from tqdm import tqdm

import numpy as np
import torch
import skimage.measure as sk_measure
import trimesh

import m01_Config_Files
import m04_DeepSDF.model_sdf as sdf_model
from m02_Data_Files.d05_SDF_Results import runs_sdf

# utils remains unchanged
from m04_DeepSDF import utils_deepsdf

# ──────────────────────────────────────────────────────────────────────────────
# Device setup
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Octree implementation
# ──────────────────────────────────────────────────────────────────────────────


class OctreeNode:
    """Simple container for one octree node."""

    __slots__ = ("center", "half", "depth", "sdf_corners")

    def __init__(self, center: np.ndarray, half: float, depth: int, sdf_corners: np.ndarray):
        self.center = center  # (3,) world‑space centre
        self.half = half  # half‑edge length
        self.depth = depth  # tree depth (root = 0)
        self.sdf_corners = sdf_corners  # (8,) SDF at the cube's corners


# Offsets to move from centre to corners (unit cube)
_CORNER_OFFSETS = torch.tensor(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.float32,
    device=device,
)


def _eval_sdf(latent: torch.Tensor, coords: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Vectorised single‑batch SDF evaluation (no grad, returns (N,))."""
    with torch.no_grad():
        latent_tile = latent.expand(coords.shape[0], -1)
        inp = torch.hstack((latent_tile, coords))
        sdf = model(inp)
    return sdf.squeeze(-1)


def _sample_cube_corners(
    latent: torch.Tensor, model: torch.nn.Module, centers: torch.Tensor, half: float
) -> np.ndarray:
    """Evaluate SDF at *eight* corners of *each* cube centre given.

    Parameters
    ----------
    latent   : (1, L) latent code on *device*
    centers  : (N, 3) cube centres on *device*
    half     : half‑edge length of *all* cubes in **world units**

    Returns
    -------
    np.ndarray of shape (N, 8)
    """
    #  shape (N*8, 3)
    corners = (centers.unsqueeze(1) + _CORNER_OFFSETS * half).view(-1, 3)
    sdf_vals = _eval_sdf(latent, corners, model)
    return sdf_vals.view(centers.shape[0], 8).cpu().numpy()


def _needs_split(sdf_vals: np.ndarray, half: float, band_width: float) -> np.ndarray:
    """Vectorised *split/non‑split* decision for each cube.

    «Split» when *either*:
      1. sign change across the eight corners, **or**
      2. ``min(|sdf|) < band_width × half × sqrt(3)`` (inside narrow band)
    """
    sign_change = (np.min(sdf_vals, axis=-1) * np.max(sdf_vals, axis=-1)) < 0.0
    close_band = np.min(np.abs(sdf_vals), axis=-1) < band_width * half * np.sqrt(3)
    return np.logical_or(sign_change, close_band)


# ──────────────────────────────────────────────────────────────────────────────
# Octree construction & rasterisation
# ──────────────────────────────────────────────────────────────────────────────


def build_octree(
    latent: torch.Tensor,
    model: torch.nn.Module,
    *,
    max_depth: int = 8,
    band_width: float = 0.5,
):
    """Return *leaf* nodes and *max_depth_used* for the adaptive octree."""
    root_half = 5.0  # bounding box is [−5, 5]³
    root_center = np.zeros(3, dtype=np.float32)
    root_sdf = _sample_cube_corners(latent, model, torch.tensor([root_center], device=device), root_half)[0]

    root = OctreeNode(root_center, root_half, 0, root_sdf)
    leaves = []
    stack = [root]

    while stack:
        node = stack.pop()

        if node.depth >= max_depth:
            leaves.append(node)
            continue

        if _needs_split(node.sdf_corners, node.half, band_width):
            child_half = node.half / 2.0
            # eight child centres (±half/2 along each axis)
            child_centers = (
                np.array([[sx, sy, sz] for sx in (-child_half, child_half) for sy in (-child_half, child_half) for sz in (-child_half, child_half)],
                         dtype=np.float32)
                + node.center
            )
            child_sdf = _sample_cube_corners(
                latent,
                model,
                torch.tensor(child_centers, device=device),
                child_half,
            )
            for cc, sdfc in zip(child_centers, child_sdf):
                stack.append(OctreeNode(cc, child_half, node.depth + 1, sdfc))
        else:
            leaves.append(node)

    return leaves, max_depth


# Pre‑computed integer corner offsets for **rasterisation**
_int_corner_offsets = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int32,
)


def octree_to_dense_grid(leaves, max_depth: int) -> np.ndarray:
    """Convert the leaf nodes into a (R+1)³ dense SDF grid where
    ``R = 2**max_depth``.

    Voxels never visited by the octree receive ``+∞`` so they are ignored by
    Marching Cubes.
    """
    res = 2**max_depth
    grid = np.full((res + 1, res + 1, res + 1), np.inf, dtype=np.float32)
    cell_size = 10.0 / res  # world size of one *unit* in the dense grid

    for node in leaves:
        # Index of the *minimum* corner of the cube in voxel space
        base_idx = ((node.center - node.half) + 5.0) / cell_size
        base_idx = base_idx.astype(np.int32)
        step = 2 ** (max_depth - node.depth)  # cube spans *step* voxels

        for offset, sdf_val in zip(_int_corner_offsets, node.sdf_corners):
            idx = tuple(base_idx + offset * step)
            # Keep the SDF with the *smaller magnitude* (closer to surface)
            if abs(sdf_val) < abs(grid[idx]):
                grid[idx] = sdf_val

    # Remaining +∞ → large positive value (far away from surface)
    grid[np.isinf(grid)] = 1.0
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# Mesh extraction helper
# ──────────────────────────────────────────────────────────────────────────────


def _extract_mesh_from_grid(grid: np.ndarray):
    """Run Marching Cubes and return (vertices, faces)."""
    verts, faces, _, _ = sk_measure.marching_cubes(grid, level=0.0)
    res = grid.shape[0] - 1  # because grid is (R+1)³
    verts = verts / res * 10.0 - 5.0  # scale back to world co‑ordinates
    return verts, faces.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction pipeline for **one** object
# ──────────────────────────────────────────────────────────────────────────────


def reconstruct_object(cfg, latent_code: torch.Tensor, obj_idx: int, obj_id_str: str, model):
    """Perform adaptive octree reconstruction and write OBJ."""

    max_depth = cfg.get("octree_max_depth", 8)
    band_width = cfg.get("octree_band_width", 0.5)

    leaves, depth_used = build_octree(latent_code, model, max_depth=max_depth, band_width=band_width)
    grid = octree_to_dense_grid(leaves, depth_used)

    try:
        vertices, faces = _extract_mesh_from_grid(grid)
    except Exception as e:
        print("Mesh extraction failed →", type(e).__name__, str(e))
        return

    mesh_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg["folder_sdf"], "meshes_training_octree")
    os.makedirs(mesh_dir, exist_ok=True)
    obj_path = os.path.join(mesh_dir, f"mesh_{obj_id_str}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type="obj")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: read **training** hyper‑parameters
# ──────────────────────────────────────────────────────────────────────────────


def _read_training_settings(cfg):
    p = os.path.join(os.path.dirname(runs_sdf.__file__), cfg["folder_sdf"], "settings.yaml")
    with open(p, "rb") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# ──────────────────────────────────────────────────────────────────────────────
# Main routine (iterates over all stored latent codes)
# ──────────────────────────────────────────────────────────────────────────────


def main(cfg):
    training_settings = _read_training_settings(cfg)

    weights_path = os.path.join(os.path.dirname(runs_sdf.__file__), cfg["folder_sdf"], "weights.pt")
    model = sdf_model.SDFModel(
        num_layers=training_settings["num_layers"],
        skip_connections=training_settings["latent_size"],
        latent_size=training_settings["latent_size"],
        inner_dim=training_settings["inner_dim"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    str2int_path = os.path.join(os.path.dirname(runs_sdf.__file__), cfg["folder_sdf"], "idx_str2int_dict.npy")
    results_path = os.path.join(os.path.dirname(runs_sdf.__file__), cfg["folder_sdf"], "results.npy")

    str2int = np.load(str2int_path, allow_pickle=True).item()
    results_dict = np.load(results_path, allow_pickle=True).item()

    for obj_id_path, obj_idx in tqdm(str2int.items(), desc="Adaptive reconstruction"):
        latent_np = results_dict["best_latent_codes"][obj_idx]
        latent_code = torch.tensor(latent_np, device=device)
        reconstruct_object(cfg, latent_code, obj_idx, obj_id_path, model)

if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), "c04_reconstruct.yaml")
    with open(cfg_path, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
