import os
import torch
import yaml

import json

from gcn_runner import GCN_Runner

import m02_Data_Files.d08_Predict_Data.d01_Config
import m02_Data_Files.d08_Predict_Data.d03_Graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inject_latent_to_json(json_folder, runner: GCN_Runner):
    # TODO: 增加对于特定节点的遮罩
    """
    Copy and inject new features to json, and save as new file.
    """
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            graph_file = os.path.join(json_folder, filename)
            new_filename = filename.replace(".json", "_prediction.json")
            new_path = os.path.join(json_folder, new_filename)

            out = runner.predict()

            with open(graph_file, "r") as f:
                data = json.load(f)

            latent_array = out.cpu().numpy()
            for i, node in enumerate(data["nodes"]):
                node["latent_code"] = latent_array[i].tolist()

            with open(new_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Latent injected and saved to: {new_filename}")


if __name__=='__main__':

    # Define paths and create folder for each new training
    Predict_data_folder = os.path.dirname(m02_Data_Files.d08_Predict_Data.d03_Graph.__file__)
    train_cfg_path = os.path.join(os.path.dirname(m02_Data_Files.d08_Predict_Data.d01_Config.__file__), 'training.yaml')

    # Load configs and training
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Init GCN Runner
    Runner = GCN_Runner(train_cfg)

    # Replace the latent in old json that represents new locatiopn
    inject_latent_to_json(Predict_data_folder, Runner)


