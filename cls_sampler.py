import sys
import os
import glob
import yaml
import trimesh

import numpy as np
import point_cloud_utils as pcu
from glob import glob
from tqdm import tqdm

import m01_Config_Files
import m02_Data_Files.d02_Object_Files
import m02_Data_Files.d04_SDF_Converted

class Sampler:
    def __init__(self):
        # Init config path
        self.cfg_path = os.path.dirname(m01_Config_Files.__file__)
        self.output_path = os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__)
        self.cfg_file_name = os.path.join(self.cfg_path, 'c01_2_extracting_sdf.yaml')      
        # Load configs
        with open(self.cfg_file_name, 'rb') as f:
            self.cfg_file = yaml.load(f, Loader=yaml.FullLoader)
        # Init all paraemters
        self._reset()

    # ---------- public API ----------
    def sample(self):
        self._load_configs(self.cfg_file)
        for obj_idx, obj_path in enumerate(tqdm(self.obj_files, desc="Processing OBJ files")):
            # Object unique index. Str to int by byte encoding
            obj_idx_str = os.path.splitext(os.path.basename(obj_path))[0] 
            self.idx_str2int_dict[obj_idx_str] = obj_idx
            self.idx_int2str_dict[obj_idx] = obj_idx_str
            # Dictionary to store the samples and SDFs
            self.samples_dict[obj_idx] = dict()
            try:
                mesh_original = trimesh.load(obj_path, force='mesh')
                if not mesh_original.is_watertight:
                    print(f"Mesh {obj_path} is not watertight, attempting to repair...")
                    mesh_original.fill_holes()
                    if not mesh_original.is_watertight:
                        print(f"Warning: Mesh {obj_path} could not be fully repaired.")
                    else:
                        print(f"Mesh {obj_path} repaired successfully.")
                self.verts = np.array(mesh_original.vertices)
                self.faces = np.array(mesh_original.faces)
                self.total_area = self._compute_triangle_areas(self.verts, self.faces).sum()
                self.volume = mesh_original.volume
            except Exception as e:
                print(f"Error processing mesh {obj_path}: {e}")

            p_total, sdf = self._sample_points_and_compute_sdf(self.verts, self.faces, self.total_area, self.volume, self.cfg_file)

            self.samples_dict[obj_idx]['sdf'] = sdf

            self.samples_dict[obj_idx]['samples_latent_class'] = self._combine_sample_latent(p_total, np.array([obj_idx], dtype=np.int32))

    # ---------- private ----------
    def _reset(self):
        # === Path ===
        self.obj_files: list[str] = []     
        self.output_path: str = ""   
        # === Data containers ===
        self.samples_dict: dict = {}
        self.idx_str2int_dict: dict = {}
        self.idx_int2str_dict: dict = {}
        self.verts: dict = {}
        self.faces: dict = {}
        # === Accumulated values ===
        self.total_area: float = 0.0
        self.volume: float = 0.0
        # === Sampling configuration ===
        self.minimal_per_face: int = 0
        self.dense_of_samples_on_surface: int = 0
        self.dense_of_samples_in_space: int = 0
        self.far_field_coefficient: float = 0.0
        self.surface_offset_1: float = 0.0
        self.surface_offset_2: float = 0.0

    def _load_configs(self, cfg_file):
        self.obj_files = glob(os.path.join(os.path.dirname(m02_Data_Files.d02_Object_Files.__file__), '*.obj'))
        self.minimal_per_face = int(cfg_file['minimal_per_surface'])
        self.dense_of_samples_on_surface = int(cfg_file['dense_of_samples_on_surface'])
        self.dense_of_samples_in_space = int(cfg_file['dense_of_samples_in_space'])
        self.far_field_coefficient = int(cfg_file['far_field_coefficient'])
        self.surface_offset_1 = int(cfg_file['surface_offset_1'])
        self.surface_offset_2 = int(cfg_file['surface_offset_2'])

    def _combine_sample_latent(self, samples, latent_class):
        latent_class_full = np.tile(latent_class, (samples.shape[0], 1))   
        return np.hstack((latent_class_full, samples))

    def _compute_triangle_areas(self, vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross_prod, axis=1)
    
    def _sample_points_and_compute_sdf(self, verts, faces, total_area, volume, cfg):

        # ---------- 1. 参数 ----------
        MIN_PER_FACE = 100
        surface_sample_num = int(float(total_area) * int(cfg['dense_of_samples_on_surface']))
        volume_sample_num  = int(float(volume)     * int(cfg['dense_of_samples_in_space']))
        far_field_sample_num = int(volume_sample_num * cfg['far_field_coefficient'])

        # ---------- 2. 初步曲面采样（按面积） ----------
        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, surface_sample_num)
        p_surf_init = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)

        # ---------- 3. 统计并补足不足 100 点的面 ----------
        face_counts = np.bincount(fid_surf, minlength=len(faces))
        deficits = np.maximum(0, MIN_PER_FACE - face_counts)  # 每面缺多少

        supplement_pts = []
        supplement_normals = []

        # 预先计算所有面的法向量，后续可直接索引
        tri_verts = verts[faces]
        face_normals_all = np.cross(tri_verts[:, 1] - tri_verts[:, 0],
                                    tri_verts[:, 2] - tri_verts[:, 0])
        face_normals_all /= np.linalg.norm(face_normals_all, axis=1, keepdims=True)

        for f_idx, n_missing in enumerate(deficits):
            if n_missing == 0:
                continue
            v0, v1, v2 = tri_verts[f_idx]
            pts = _sample_points_on_face(v0, v1, v2, n_missing)
            supplement_pts.append(pts)
            # 为补充点附加对应面的法向量
            supplement_normals.append(np.repeat(face_normals_all[f_idx:f_idx+1],
                                                n_missing, axis=0))

        if supplement_pts:   # 若确实补了点
            p_surf_extra = np.vstack(supplement_pts)
            normals_extra = np.vstack(supplement_normals)
            # 将初步采样得到的法向量取出来
            normals_init = face_normals_all[fid_surf]
            # 合并表面点与法向量
            p_surf = np.vstack((p_surf_init, p_surf_extra))
            face_normals = np.vstack((normals_init, normals_extra))
        else:
            p_surf = p_surf_init
            face_normals = face_normals_all[fid_surf]

        # ---------- 4. 生成 Offset 点 ----------
        offset_distance_1 = cfg['surface_offset_1']
        offset_distance_2 = cfg['surface_offset_2']
        offset_distance_3 = offset_distance_1 / 2.0

        p_surf_out_1 = p_surf + offset_distance_1 * face_normals
        p_surf_out_2 = p_surf + offset_distance_2 * face_normals
        p_surf_in    = p_surf - offset_distance_3 * face_normals

        # ---------- 5. 体积随机点 ----------
        centroid = np.mean(verts, axis=0)
        centered_verts = verts - centroid
        cov = np.cov(centered_verts, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)

        local_verts = centered_verts @ eig_vecs
        local_min = local_verts.min(axis=0)
        local_max = local_verts.max(axis=0)
        local_center = (local_min + local_max) / 2.0
        half_range = (local_max - local_min) / 2.0
        new_half_range = half_range + 0.5
        new_local_min = local_center - new_half_range
        new_local_max = local_center + new_half_range

        p_vol_local = np.random.uniform(low=new_local_min,
                                        high=new_local_max,
                                        size=(volume_sample_num, 3))
        p_vol = p_vol_local @ eig_vecs.T + centroid

        # ---------- 6. 远场点 ----------
        diag = np.linalg.norm(local_max - local_min)
        def Sample_on_sphere_surface(center, radius, n_pts):
            """在给定球面上均匀采样 n_pts 个点（局部坐标）。"""
            # Marsaglia (1972) 方法
            xyz = np.random.normal(size=(n_pts, 3))
            xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
            return center + radius * xyz

        p_far_local_1 = Sample_on_sphere_surface(local_center, diag + 5,  far_field_sample_num)
        p_far_local_2 = Sample_on_sphere_surface(local_center, diag + 10, far_field_sample_num)
        p_far_local_3 = Sample_on_sphere_surface(local_center, diag + 20, far_field_sample_num)

        p_far_1 = p_far_local_1 @ eig_vecs.T + centroid
        p_far_2 = p_far_local_2 @ eig_vecs.T + centroid
        p_far_3 = p_far_local_3 @ eig_vecs.T + centroid

        # ---------- 7. 汇总并计算 SDF ----------
        p_total = np.vstack((
            p_vol,
            p_surf_out_1, p_surf_out_2, p_surf_in,
            p_surf,
            p_far_1, p_far_2, p_far_3
        ))
        sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)

        return p_total, sdf



