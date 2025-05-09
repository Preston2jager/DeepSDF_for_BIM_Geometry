import os
import json
import torch
import dgl
import yaml

from dgl.data import DGLDataset

import m02_Data_Files.d06_GCN_Training.d01_Graphs
import m02_Data_Files.d06_GCN_Training.d02_Configs
import m02_Data_Files.d08_Predict_Data.d01_Config
import m02_Data_Files.d08_Predict_Data.d03_Graph

class GCNDataset(DGLDataset):

    def __init__(self):
        super().__init__(name='GCN_dataset')
  
    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
    
    def load_data(self, graph_folder, cfg_folder):
        """
        General function to load json graph file and construct it into dgl graph
        """
        # Loading yaml to get ifcclass
        ifc_cfg_file = os.path.join(cfg_folder, 'ifc.yaml')
        with open(ifc_cfg_file, 'rb') as f:
            data_cfg = yaml.load(f, Loader=yaml.FullLoader)
        ifc_classes = data_cfg['ifc_classes']
        type_to_idx = {cls_name: idx for idx, cls_name in enumerate(ifc_classes)}
        num_classes = len(ifc_classes)
        # Loading all jsons to create graph, each json represent one graph.
        graph_files = [
            f for f in os.listdir(graph_folder)
            if f.endswith(".json")
        ]
        self.graphs = []
        for file_name in graph_files:
            # Get data from json
            with open(os.path.join(graph_folder, file_name), 'r') as f:
                data = json.load(f)
            # Nodes
            node_features = []
            targets = []
            nodes_sorted = sorted(data['nodes'], key=lambda x: x['index'])
            for node in nodes_sorted:
                type_name = node['type']
                type_idx = type_to_idx[type_name]
                one_hot_vector = torch.nn.functional.one_hot(torch.tensor(type_idx), num_classes=num_classes).float()
                latent_code = torch.tensor(node['latent_code'], dtype=torch.float32)
                full_feature = torch.cat([one_hot_vector, latent_code], dim=0)
                node_features.append(full_feature)
                targets.append(latent_code)
            # Feature list convert
            node_features = torch.stack(node_features)
            targets = torch.stack(targets) 
            # Edges
            edges = data['edges']
            src, dst = zip(*edges)
            src = torch.tensor(src)
            dst = torch.tensor(dst)
            g = dgl.graph((src, dst))
            g = dgl.to_bidirected(g)
            g.ndata['feat'] = node_features
            g.ndata['target'] = targets 
            # Adding graph
            self.graphs.append(g)
        self.batched_graph = dgl.batch(self.graphs)
    
    def load_training(self):
        """
        Using configs and data in training folder to create training dataset.
        """
        cfg_folder = os.path.dirname(m02_Data_Files.d06_GCN_Training.d02_Configs.__file__)
        graph_folder = os.path.dirname(m02_Data_Files.d06_GCN_Training.d01_Graphs.__file__)
        self.load_data(graph_folder, cfg_folder)

    def load_predict(self):
        """
        Using configs and data in predict folder to create training dataset.
        """
        cfg_folder = os.path.dirname(m02_Data_Files.d08_Predict_Data.d01_Config.__file__)
        graph_folder = os.path.dirname(m02_Data_Files.d08_Predict_Data.d03_Graph.__file__)
        self.load_data(graph_folder, cfg_folder)

def main():
    data = GCNDataset()
    print(data.__getitem__(0))

if __name__=='__main__':
    main()