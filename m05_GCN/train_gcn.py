import os
import yaml 
import torch
import shutil

from datetime import datetime
from gcn_runner import GCN_Runner

import m01_Config_Files
import m02_Data_Files.d06_GCN_Training
import m02_Data_Files.d07_GCN_Results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def copy_all(folder_path, target_dir):
    """
    Copy all files including files in sub-folder
    """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            source_file = os.path.join(root, filename)
            rel_path = os.path.relpath(source_file, folder_path)
            target_file = os.path.join(target_dir, rel_path)
            target_subdir = os.path.dirname(target_file)
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)
            shutil.copy2(source_file, target_file)
            print(f"Copied: {source_file} -> {target_file}")

def create_training_folder():
    """
    Create a folder based on current time stamp
    """
    timestamp_run = datetime.now().strftime('%d_%m_%H%M%S')
    runs_dir_path = os.path.dirname(m02_Data_Files.d07_GCN_Results.__file__)
    run_dir = os.path.join(runs_dir_path, timestamp_run)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

if __name__=='__main__':

    # Define paths and create folder for each new training
    data_folder = os.path.dirname(m02_Data_Files.d06_GCN_Training.__file__)
    target_folder = create_training_folder()
    train_cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), 'training.yaml')
    output_weight_file = os.path.join(target_folder, "best_model.pth")

    # Copy all training data into the new training folder for reference
    copy_all(data_folder, target_folder)

    # Load configs and training
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    Runner = GCN_Runner(train_cfg, output_weight_file)
    Runner.train() 
