import os
import time 
import torch

from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

import m05_GCN.model_gcn  as model_gcn
from m05_GCN.dataset_gcn import GCNDataset

import m02_Data_Files.d07_GCN_Results
import m02_Data_Files.d08_Predict_Data.d01_Config
import m02_Data_Files.d08_Predict_Data.d05_GCN


class GCN_Runner:

    def __init__(self, config, output_weight_file="None"):
        if output_weight_file == "None":
            print("Output_path not given, predict only")
        self.cfg = config
        self.output = output_weight_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.model = model_gcn.GCN(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg['GCN_lr_model'],
            weight_decay=5e-4
        )

    def get_loaders(self):
        """
        Init for training dataloader.
        """
        dataset = GCNDataset()
        dataset.load_training()
        train_size = int(len(dataset) * 0.7)
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = GraphDataLoader(
            train_data,
            batch_size=self.cfg['GCN_batch_size'],
            shuffle=True
        )
        val_loader = GraphDataLoader(
            val_data,
            batch_size=self.cfg['GCN_batch_size'],
            shuffle=False
        )
        return train_loader, val_loader
    
    def get_predict_loader(self):
        """
        Load a single graph for predicting.
        """
        dataset = GCNDataset()
        dataset.load_predict()
        dataset_size = len(dataset)
        batch = min(dataset_size, self.cfg['GCN_batch_size'])
        predict_loader = GraphDataLoader(
            dataset,
            batch_size = batch,
            shuffle = False
        )   
        return predict_loader

    def train(self):
        output_weight_file = self.output
        train_loader, val_loader = self.get_loaders()
        start_time = time.time()
        best_val_loss = float('inf')
        for epoch in range(self.cfg['GCN_epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.val_epoch(val_loader)

            print(f"Epoch [{epoch+1}/{self.cfg['GCN_epochs']}], "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), output_weight_file)
                print(f"Best model saved at epoch {epoch+1} with Val Loss {val_loss:.4f}")
        print(f"Training completed in {time.time() - start_time:.2f} s")

    def train_epoch(self, loader):
        mask = int(self.cfg['GCN_tag_dim'])
        self.model.train()
        total_loss = 0
        for batched_graph in loader:
            batched_graph = batched_graph.to(self.device)
            x = batched_graph.ndata['feat']           
            target = batched_graph.ndata['target']   
            num_nodes = batched_graph.number_of_nodes()
            per_graph_loss = 0
            for i in range(num_nodes):
                x_masked = x.clone()
                x_masked[i, mask:] = 0
                out = self.model(batched_graph, x_masked)
                pred = out[i].unsqueeze(0)      # [1, latent_dim]
                true = target[i].unsqueeze(0)   # [1, latent_dim]
                loss = self.balanced_abs_loss(pred, true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                per_graph_loss += loss.item()
            total_loss += per_graph_loss / num_nodes
        return total_loss / len(loader)

    def val_epoch(self, loader):
        mask = int(self.cfg['GCN_tag_dim'])
        self.model.eval()
        total_loss = 0
        with torch.no_grad():  
            for batched_graph in loader:
                batched_graph = batched_graph.to(self.device)
                x = batched_graph.ndata['feat']
                target = batched_graph.ndata['target']
                num_nodes = batched_graph.number_of_nodes()
                per_graph_loss = 0
                for i in range(num_nodes):
                    x_masked = x.clone()
                    x_masked[i, mask:] = 0
                    out = self.model(batched_graph, x_masked)
                    pred = out[i].unsqueeze(0)
                    true = target[i].unsqueeze(0)
                    loss = self.balanced_abs_loss(pred, true)
                    per_graph_loss += loss.item()
                total_loss += per_graph_loss / num_nodes
        return total_loss / len(loader)

    
    def predict(self):
        #TODO: 增加对于特定节点的遮罩
        self.model.eval()
        # Load weights
        weight_folder_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d05_GCN.__file__)
        weight_file = os.path.join(weight_folder_path, "best_model.pth")
        self.model.load_state_dict(torch.load(weight_file, map_location=self.device))
        # Load graphs
        predict_loader = self.get_predict_loader()
        # Predict
        predictions = []
        with torch.no_grad():
            for batched_graph in predict_loader:
                batched_graph = batched_graph.to(self.device)
                x = batched_graph.ndata['feat']
                x_masked = x.clone()
                x_masked[:, 512:] = 0
                out = self.model(batched_graph, x_masked)
                predictions.append(out.cpu())  # 保留在 CPU 上做后处理

        return torch.cat(predictions, dim=0)  # 拼成一个完整的 [N, D] 输出

    
    def balanced_abs_loss(self, pred, target, beta=0.3):
        """
        beta∈[0,1] 越大越关注维度之间的均衡性
        """
        per_sample_err = (pred - target).abs()
        mean_loss      = per_sample_err.mean()
        dim_mean_err   = per_sample_err.mean(dim=0)
        var_loss       = dim_mean_err.var()
        return (1 - beta) * mean_loss + beta * var_loss

    def topk_dim_loss(self, pred, target, k=32):
        """
        每个 batch 里：对 512 维逐维平均 |误差|，取最大的 k 维做均值。
        """
        per_sample_err = (pred - target).abs()
        dim_mean_err   = per_sample_err.mean(dim=0)
        topk_vals, _   = torch.topk(dim_mean_err, k)
        return topk_vals.mean()

    def focal_l1_loss(self, pred, target, gamma=2.0):
        """
        gamma>0；gamma 越大，越强调大误差样本/维度
        """
        err = (pred - target).abs()
        weights = (err / (err.detach().mean() + 1e-8)) ** gamma
        loss = (weights * err).mean()
        return loss