import os
import datetime
import shutil

import m02_Data_Files.d07_GCN_Results
import m02_Data_Files.d08_Predict_Data.d01_Config
import m02_Data_Files.d08_Predict_Data.d03_Graph
import m02_Data_Files.d08_Predict_Data.d04_SDF
import m02_Data_Files.d08_Predict_Data.d05_GCN

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
    GCN_results_folder_path = os.path.dirname(m02_Data_Files.d07_GCN_Results.__file__)

    # Define target folder pathes
    Target_CONFIG_folder_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d01_Config.__file__)
    Target_GRAPH_folder_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d03_Graph.__file__)
    Target_SDF_folder_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d04_SDF.__file__)
    Target_GCN_folder_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d05_GCN.__file__)
    
    # Delete old files
    file_types=[".pt",".0",".npy",".yaml",".json"]
    for type in file_types:
        # Clear graph folder
        Delete_files(Target_CONFIG_folder_path, type)
        Delete_files(Target_GRAPH_folder_path, type)
        Delete_files(Target_SDF_folder_path, type)
        Delete_files(Target_GCN_folder_path, type)
    print("Old files deleted")

    nearest_folder = find_nearest_folder(GCN_results_folder_path)

    if nearest_folder:
        print(f"Use：{nearest_folder}")
        # All basic configs
        Source_CONFIG_folder_path = os.path.join(nearest_folder,"d02_Configs")
        copy_files_from_folder(Source_CONFIG_folder_path, Target_CONFIG_folder_path)
        # Only weights.pt for new obj file
        Source_SDF_folder_path = os.path.join(nearest_folder,"d03_SDF_Latent_Codes")
        Source_SDF_weight_file = os.path.join(Source_SDF_folder_path, "weights.pt")
        shutil.copy2(Source_SDF_weight_file, Target_SDF_folder_path)
        # GCN weight on root path
        Source_GCN_weight_file  = os.path.join(nearest_folder,"best_model.pth") 
        shutil.copy2(Source_GCN_weight_file, Target_GCN_folder_path)
        print("=============================================================")
        print("SDF and GCN weights and configs ready in Predict_Data folder")
        print("=============================================================")

if __name__=='__main__':
    main()

