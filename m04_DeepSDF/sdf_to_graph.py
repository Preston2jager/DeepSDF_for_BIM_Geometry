import os
import datetime
import shutil
import json
import numpy as np

import m01_Config_Files
import m02_Data_Files.d03_Graph
import m02_Data_Files.d06_GCN_Training.d01_Graphs
import m02_Data_Files.d05_SDF_Results.runs_sdf
import m02_Data_Files.d06_GCN_Training.d02_Configs
import m02_Data_Files.d06_GCN_Training.d03_SDF_Latent_Codes

def parse_folder_time(folder_name):
    try:
        parts = folder_name.split('_')
        if len(parts) != 3:
            raise ValueError(f"Incorrect folder name format: {folder_name}")
        day = int(parts[0])
        month = int(parts[1])
        hour = int(parts[2][0:2])
        minute = int(parts[2][2:4])
        second = int(parts[2][4:6])
        year = datetime.datetime.now().year
        return datetime.datetime(year, month, day, hour, minute, second)
    except Exception as e:
        print(f"Failed to extract：{folder_name}， Error：{e}")
        return None
    
def find_nearest_folder(base_dir):
    """Find the lastest folder"""
    now = datetime.datetime.now()
    folders = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            folder_time = parse_folder_time(name)
            if folder_time:
                folders.append((folder_time, full_path))
    if not folders:
        print("No valid folder found")
        return None

    folders.sort(key=lambda x: abs((x[0] - now).total_seconds()))
    return folders[0][1]

def copy_files_from_folder(folder_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(folder_path):
        source_file = os.path.join(folder_path, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_dir)
            print(f"Copyed：{source_file}")

def adding_latent():
    sdf_base_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d03_SDF_Latent_Codes.__file__)
    graph_base_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d01_Graphs.__file__)
    idx_file = os.path.join(sdf_base_path, "idx_int2str_dict.npy")
    latent_file = os.path.join(sdf_base_path, "results.npy") 

    idx2name = np.load(idx_file, allow_pickle=True).item()
    latent_codes_raw = np.load(latent_file, allow_pickle=True)
    latent_codes_dict = latent_codes_raw.item()  
    latent_codes = latent_codes_dict["best_latent_codes"]

    if len(idx2name) != len(latent_codes):
        raise ValueError(f"Length of idx2name({len(idx2name)})does not match({len(latent_codes)})")
    name_to_latent = {}

    for idx, full_name in idx2name.items():
        if '_' in full_name:
            parts = full_name.split('_', 2)  
            if len(parts) == 3:
                global_id = parts[2]  
                name_to_latent[global_id] = latent_codes[idx]
            else:
                print(f"Fail to extract from {full_name}")

    for filename in os.listdir(graph_base_path):
        if filename.endswith(".json"):
            filepath = os.path.join(graph_base_path, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                global_id = node.get("GlobalId")
                #type_ = node.get("type")
                key = f"{global_id}"
                if key in name_to_latent:
                    latent_vector = name_to_latent[key]
                    node["latent_code"] = latent_vector.tolist()  # 转成list，json不能存numpy array
                else:
                    print(f"Fail to find {key}， Skip")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated：{filename}")

def Delete_files(folder_path, file_type):
    """
    Delete all files of a type in a folder.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_type):
            target_file_path = os.path.join(folder_path, filename)
            os.remove(target_file_path)
            print(f"File Deleted: {target_file_path}")

def main():
    # Define source folder pathes
    Source_config_folder_path = os.path.dirname(m01_Config_Files.__file__)
    Source_graph_folder_path = os.path.dirname(m02_Data_Files.d03_Graph.__file__)
    Source_sdf_results_folder_path = os.path.dirname(m02_Data_Files.d05_SDF_Results.runs_sdf.__file__)
    # Define target folder pathes
    Target_graph_folder_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d01_Graphs.__file__)
    Target_configs_folder_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d02_Configs.__file__)
    Target_sdf_folder_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d03_SDF_Latent_Codes.__file__)
    # Delete old files
    file_types=[".pt",".0",".npy",".yaml",".json"]
    #for type in file_types:
        #Delete_files(Target_graph_folder_path, type)
        #Delete_files(Target_configs_folder_path, type)
        #Delete_files(Target_sdf_folder_path, type)
    #print("Old files deleted")

    nearest_folder = find_nearest_folder(Source_sdf_results_folder_path)
    if nearest_folder:
        print(f"Use：{nearest_folder}")
        copy_files_from_folder(nearest_folder, Target_sdf_folder_path)
        copy_files_from_folder(Source_graph_folder_path, Target_graph_folder_path)
        adding_latent()
        copy_files_from_folder(Source_config_folder_path, Target_configs_folder_path)

if __name__=='__main__':
    main()

