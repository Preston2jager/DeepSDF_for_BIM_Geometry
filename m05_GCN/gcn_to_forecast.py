import os
import shutil
import json
import numpy as np

import m01_Config_Files
import m02_Data_Files.d03_Graph
import m02_Data_Files.d06_GCN_Training.d01_Graphs
import m02_Data_Files.d05_SDF_Results.runs_sdf
import m02_Data_Files.d06_GCN_Training.d02_Configs
import m02_Data_Files.d06_GCN_Training.d03_SDF_Latent_Codes

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
    Source_config_folder_path = os.path.dirname(m01_Config_Files.__file__)
    Source_graph_folder_path = os.path.dirname(m02_Data_Files.d03_Graph.__file__)
    Source_sdf_results_folder_path = os.path.dirname(m02_Data_Files.d05_SDF_Results.runs_sdf.__file__)
    # Define target folder pathes
    Target_graph_folder_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d01_Graphs.__file__)
    Target_configs_folder_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d02_Configs.__file__)
    Target_sdf_folder_path = os.path.dirname(m02_Data_Files.d06_GCN_Training.d03_SDF_Latent_Codes.__file__)
    # Delete old files
    file_types=[".pt",".0",".npy",".yaml",".json"]
    for type in file_types:
        Delete_files(Target_graph_folder_path, type)
        Delete_files(Target_configs_folder_path, type)
        Delete_files(Target_sdf_folder_path, type)
    print("Old files deleted")
    # Copy
    copy_files_from_folder(nearest_folder, Target_sdf_folder_path)
    copy_files_from_folder(Source_graph_folder_path, Target_graph_folder_path)
    adding_latent()
    copy_files_from_folder(Source_config_folder_path, Target_configs_folder_path)

if __name__=='__main__':
    main()

