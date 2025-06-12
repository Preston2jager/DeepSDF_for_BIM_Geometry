import ifcopenshell
import yaml
import os
import sys
import argparse

from utils_data_preprocessing import *

import m01_Config_Files
import m02_Data_Files.d01_Raw_IFC
import m02_Data_Files.d01_Raw_IFC.d01_Expanded
import m02_Data_Files.d02_Object_Files
import m02_Data_Files.d04_SDF_Converted


def main(cfg,ifc_cfg,args):
    # Extracting objects from IFC files
    # Define paths
    if args.mode == "train":
        cfg_folder_path = os.path.dirname(m01_Config_Files.__file__)
        IFC_folder_path = os.path.dirname(m02_Data_Files.d01_Raw_IFC.__file__)
        Expanded_ifc_folder_path = os.path.dirname(m02_Data_Files.d01_Raw_IFC.d01_Expanded.__file__)
        Object_folder_path = os.path.dirname(m02_Data_Files.d02_Object_Files.__file__)
        NPY_folder_path = os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__)
    # Delete previous files
    Delete_files(Expanded_ifc_folder_path, ".ifc")
    Delete_files(Object_folder_path, ".obj")
    Delete_files(NPY_folder_path, ".npy")
    # Copy raw IFC files into expanded folders.
    for filename in os.listdir(IFC_folder_path):
        if filename.lower().endswith('.ifc'):
            Source_file_path = os.path.join(IFC_folder_path, filename)
            Target_file_path = os.path.join(Expanded_ifc_folder_path, filename)
            shutil.copy(Source_file_path, Target_file_path)
    # Expand files if required
    if bool(cfg.get("data_expand")) is True:
        Num_of_copy = cfg.get("copies")
        for filename in os.listdir(IFC_folder_path):
            if filename.lower().endswith('.ifc'):
                IFC_file_expand(filename, Num_of_copy, IFC_folder_path, Expanded_ifc_folder_path)
    # Generate new global id for all files
    for filename in os.listdir(Expanded_ifc_folder_path):
        if filename.lower().endswith('.ifc'):
            Regenerate_global_ids(filename, Expanded_ifc_folder_path)
    # Export IFCs to .obj files 
    index = 1
    settings = ifcopenshell.geom.settings() 
    settings.set(settings.USE_WORLD_COORDS, True)
    IFC_classes = ifc_cfg.get("ifc_classes", [])
    for filename in os.listdir(Expanded_ifc_folder_path):
        if filename.lower().endswith('.ifc'):
            IFC_file_path = os.path.join(Expanded_ifc_folder_path, filename)
            IFC_file = ifcopenshell.open(IFC_file_path)
            IFC_to_obj(Object_folder_path, IFC_file, IFC_classes, index, settings) 
            index += 1
    # Randomlise expanded .obj files 
    Batch_obj_transform(Object_folder_path,cfg)
        
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Mode of operation, train or pred", nargs='?', default="train")
    args = parser.parse_args()

    if args.mode == "train":
        print("Extract for training data")
        cfg_path = os.path.dirname(m01_Config_Files.__file__)
        cfg_file = os.path.join(cfg_path, 'c01_extracting.yaml')
        ifc_cfg_file = os.path.join(cfg_path, 'c02_ifc.yaml')
    elif args.mode == "pred":
        print("Extract for prediction data")
        cfg_path = os.path.dirname(m02_Data_Files.d01_Config.__file__)
        cfg_file = os.path.join(cfg_path, 'extracting.yaml')
        ifc_cfg_file = os.path.join(cfg_path, 'ifc.yaml')
    else:
        print("Incorrect mode, train or pred?")
        sys.exit(1)

    with open(cfg_file, 'rb') as f:
        cfg_yaml = yaml.load(f, Loader=yaml.FullLoader)
    with open(ifc_cfg_file, 'rb') as f:
        ifc_yaml = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg_yaml, ifc_yaml, args)
