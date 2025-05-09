import torch
import os
import sys
import yaml
import argparse

import numpy as np
from tqdm import tqdm

from sdf_runner import SDF_Runner

import model_sdf as sdf_model
import dataset_sdf as dataset
import m01_Config_Files
import m02_Data_Files.d04_SDF_Converted
import m02_Data_Files.d05_SDF_Results.runs_sdf as runs
import m02_Data_Files.d08_Predict_Data.d01_Config
import m02_Data_Files.d08_Predict_Data.d04_SDF

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

if __name__=='__main__':
    torch.cuda.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Mode of operation, train or pred", nargs='?', default="train")
    args = parser.parse_args()

    if args.mode == "train":
        print("Extract for training data")
        data_folder_path = os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__)
        train_cfg_path = os.path.dirname(m01_Config_Files.__file__)
        train_cfg_file = os.path.join(train_cfg_path, 'training.yaml')
        Target_folder_path = os.path.dirname(runs.__file__)
    elif args.mode == "pred":
        print("Extract for prediction data")
        train_cfg_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d01_Config.__file__)
        train_cfg_file = os.path.join(train_cfg_path, 'training.yaml')
        data_folder_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d04_SDF.__file__)
    else:
        print("Incorrect mode, train or pred?")
        sys.exit(1)

    with open(train_cfg_file, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    Runner = SDF_Runner(train_cfg, args)

    if args.mode == "train":
        Runner.train_standard(data_folder_path)
    elif args.mode == "pred":
        Runner.train_latent_only(data_folder_path)