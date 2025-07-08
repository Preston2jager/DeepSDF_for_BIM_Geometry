import torch
import os
import trimesh
import yaml
import numpy as np
from tqdm import tqdm

import m01_Config_Files
import m04_DeepSDF.model_sdf as sdf_model
from m04_DeepSDF import utils_deepsdf
from m02_Data_Files.d05_SDF_Results import runs_sdf

"""Extract mesh from an already optimised latent code and network. 
Store the mesh in the same folder where the latent code is located."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_params(cfg):
    """Read the settings from the settings.yaml file. These are the settings used during training."""
    training_settings_path = os.path.join(os.path.dirname(runs_sdf.__file__),  cfg['folder_sdf'], 'settings.yaml') 
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)
    return training_settings


def reconstruct_object(cfg, latent_code, obj_idx, obj_id_str, model, coords_batches, grad_size_axis): 
    """
    Reconstruct the object from the latent code and save the mesh.
    Meshes are stored as .obj files under the same folder cerated during training, for example:
    - runs_sdf/<datetime>/meshes_training/mesh_0.obj
    """
    sdf = utils_deepsdf.predict_sdf(latent_code, coords_batches, model)
    try:
        vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
    except Exception as e:
        print('Mesh extraction failed')
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        return
    
    # save mesh as obj
    mesh_dir = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'meshes_training')
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)
    obj_path = os.path.join(mesh_dir, f"mesh_{obj_id_str}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')


def main(cfg):
    training_settings = read_params(cfg)
    # Load the model
    weights = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'weights.pt')
    model = sdf_model.SDFModel(
        num_layers=training_settings['num_layers'], 
        skip_connections=training_settings['latent_size'], 
        latent_size=training_settings['latent_size'], 
        inner_dim=training_settings['inner_dim']).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
   
    # Extract mesh obtained with the latent code optimised at inference
    coords, grad_size_axis = utils_deepsdf.get_volume_coords(cfg['resolution'])
    coords = coords.to(device)

    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)
    
    # Load paths
    str2int_path = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'idx_str2int_dict.npy')
    results_dict_path = os.path.join(os.path.dirname(runs_sdf.__file__), cfg['folder_sdf'], 'results.npy')
    
    # Load dictionaries
    str2int_dict = np.load(str2int_path, allow_pickle=True).item()
    results_dict = np.load(results_dict_path, allow_pickle=True).item()

    for obj_id_path, obj_idx in tqdm(str2int_dict.items(), desc="Processing Objects"):
    # 从结果字典中获取训练过程中优化得到的 latent code
        latent_code = results_dict['best_latent_codes'][obj_idx]
        latent_code = torch.tensor(latent_code).to(device)
        reconstruct_object(cfg, latent_code, obj_idx, obj_id_path, model, coords_batches, grad_size_axis)
    
    


if __name__ == '__main__':
    cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), 'c04_reconstruct.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)