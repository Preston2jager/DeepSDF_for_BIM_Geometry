import torch
import os
import time
import yaml

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
from tqdm import tqdm

from utils.utils_deepsdf import SDFLoss_multishape_full_exp
from utils import utils_deepsdf

import model_sdf as sdf_model
import dataset_sdf as dataset
import m01_Config_Files
import m02_Data_Files.d04_SDF_Converted
import m02_Data_Files.d05_SDF_Results.runs_sdf as runs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

class Trainer():
    
    def __init__(self, train_cfg):

        # Directories and paths
        self.train_cfg = train_cfg
        self.timestamp_run = datetime.now().strftime('%d_%m_%H%M%S')   # timestamp to use for logging data
        self.runs_dir = os.path.dirname(runs.__file__)               # directory fo all runs
        self.run_dir = os.path.join(self.runs_dir, self.timestamp_run)  # directory for this run
        os.makedirs(self.run_dir, exist_ok=True)

        # Copy index files
        Source_idx_str2int_path = os.path.join(os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__), 'idx_str2int_dict.npy')
        Target_idx_str2int_path = os.path.join(self.run_dir, 'idx_str2int_dict.npy')
        if os.path.exists(Source_idx_str2int_path):
            idx_str2int_dict = np.load(Source_idx_str2int_path, allow_pickle=True).item()
            np.save(Target_idx_str2int_path, idx_str2int_dict)
            print(f'Saved idx_str2int_dict to {Target_idx_str2int_path}')
        else:
            print(f'Warning: {Source_idx_str2int_path} not found! idx_str2int_dict not saved.')

        Source_idx_int2str_path = os.path.join(os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__), 'idx_int2str_dict.npy')
        Target_idx_int2str_path = os.path.join(self.run_dir, 'idx_int2str_dict.npy')
        if os.path.exists(Source_idx_int2str_path):
            idx_int2str_dict = np.load(Source_idx_int2str_path, allow_pickle=True).item()
            np.save(Target_idx_int2str_path, idx_int2str_dict)
            print(f'Saved idx_int2str_dict to {Target_idx_int2str_path}')
        else:
            print(f'Warning: {Source_idx_int2str_path} not found! idx_str2int_dict not saved.')

        # Logging
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_path = os.path.join(self.run_dir, 'settings.yaml')
        with open(self.log_path, 'w') as f:
            yaml.dump(self.train_cfg, f)

        # Instantiate model and optimisers
        self.model = sdf_model.SDFModel(
                self.train_cfg['num_layers'], 
                self.train_cfg['skip_connections'], 
                inner_dim=self.train_cfg['inner_dim'],
                latent_size=self.train_cfg['latent_size']
            ).float().to(device)
        
        # Define optimisers
        self.optimizer_model = optim.Adam(self.model.parameters(), lr=self.train_cfg['lr_model'], weight_decay=0)

        # Calculate num objects in samples_dictionary, wich is the number of keys
        samples_dict_path = os.path.join(os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__), f'samples_dict.npy')
        samples_dict = np.load(samples_dict_path, allow_pickle=True).item()

        # generate a unique random latent code for each shape
        self.latent_codes = utils_deepsdf.generate_latent_codes(self.train_cfg['latent_size'], samples_dict)
        self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)

        # Load pretrained weights and optimisers to continue training
        if self.train_cfg['pretrained']:
            # load pretrained weights
            self.model.load_state_dict(torch.load(self.train_cfg['pretrain_weights'], map_location=device))
            # load pretrained optimisers
            self.optimizer_model.load_state_dict(torch.load(self.train_cfg['pretrain_optim_model'], map_location=device))
            # retrieve latent codes from results.npy file
            results_path = self.train_cfg['pretrain_optim_model'].split(os.sep)
            results_path[-1] = 'results.npy'
            results_path = os.sep.join(results_path)
            # load latent codes from results.npy file
            results_latent_codes = np.load(results_path, allow_pickle=True).item()
            self.latent_codes = torch.tensor(results_latent_codes['best_latent_codes']).float().to(device)
            self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)
            self.optimizer_latent.load_state_dict(torch.load(self.train_cfg['pretrain_optim_latent'], map_location=device))

        if self.train_cfg['lr_scheduler']:
            self.scheduler_model =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')
            self.scheduler_latent =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_latent, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')

    def __call__(self):
        # TODO: change the structure
        # get data
        train_loader, val_loader = self.get_loaders()
        self.results = {
            'best_latent_codes' : []
        }

        best_loss = 10000000000
        start = time.time()
        for epoch in range(self.train_cfg['epochs']):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch
            avg_train_loss = self.train(train_loader)
            with torch.no_grad():
                avg_val_loss = self.validate(val_loader)
                epoch_end = time.time()

                if avg_val_loss < best_loss:
                    best_loss = np.copy(avg_val_loss)
                    best_weights = self.model.state_dict()
                    best_latent_codes = self.latent_codes.detach().cpu().numpy()
                    optimizer_model_state = self.optimizer_model.state_dict()
                    optimizer_latent_state = self.optimizer_latent.state_dict()

                    np.save(os.path.join(self.run_dir, 'results.npy'), self.results)
                    torch.save(best_weights, os.path.join(self.run_dir, 'weights.pt'))
                    torch.save(optimizer_model_state, os.path.join(self.run_dir, 'optimizer_model_state.pt'))
                    torch.save(optimizer_latent_state, os.path.join(self.run_dir, 'optimizer_latent_state.pt'))
                    self.results['best_latent_codes'] = best_latent_codes

                if self.train_cfg['lr_scheduler']:
                    self.scheduler_model.step(avg_val_loss)
                    self.scheduler_latent.step(avg_val_loss)
                    self.writer.add_scalar('Learning rate (model)', self.scheduler_model._last_lr[0], epoch)
                    self.writer.add_scalar('Learning rate (latent)', self.scheduler_latent._last_lr[0], epoch)            
            
        end = time.time()
        print(f'Time elapsed: {end - start} s')

    def get_loaders(self):

        data = dataset.SDFDataset()

        if self.train_cfg['clamp']:
            data.data['sdf'] = torch.clamp(data.data['sdf'], -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])

        train_size = int(0.7 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(data, [train_size, val_size])
        train_loader = DataLoader(
                train_data,
                batch_size=self.train_cfg['batch_size'],
                shuffle=True,
                num_workers=8,  
                drop_last=True,
                pin_memory=True  
            )
        val_loader = DataLoader(
            val_data,
            batch_size=self.train_cfg['batch_size'],
            shuffle=False,
            num_workers=8,
            drop_last=True,
            pin_memory=True  
            )
        return train_loader, val_loader

    def generate_xy(self, batch):
        """
        Combine latent code and coordinates.
        Return:
            - x: latent codes + coordinates, torch tensor shape (batch_size, latent_size + 3)
            - y: ground truth sdf, shape (batch_size, 1)
            - latent_codes_indices_batch: all latent class indices per sample, shape (batch size, 1).
                                            e.g. [[2], [2], [1], ..] eaning the batch contains the 2nd, 2nd, 1st latent code
            - latent_batch_codes: all latent codes per sample, shape (batch_size, latent_size)
        Return ground truth as y, and the latent codes for this batch.
        """
        batch[0] = batch[0].to(device,non_blocking=True)  
        batch[1] = batch[1].to(device,non_blocking=True)  

        latent_classes_batch = batch[0][:, 0].view(-1, 1).to(torch.long)              
        coords = batch[0][:, 1:]       
      
        latent_codes_batch = self.latent_codes[latent_classes_batch.view(-1)]         

        x = torch.hstack((latent_codes_batch, coords))                 
        y = batch[1]     # (batch_size, 1)

        return x, y, latent_classes_batch.view(-1), latent_codes_batch

    def train(self, train_loader):
        total_loss = 0.0
        iterations = 0.0
        self.model.train()

        #for batch in train_loader:
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            start_time = time.perf_counter()

            iterations += 1.0
            self.optimizer_model.zero_grad()
            self.optimizer_latent.zero_grad()

            x, y, latent_codes_indices_batch, latent_codes_batch = self.generate_xy(batch)

            x = x.to(device)  
            y = y.to(device)

            latent_codes_indices_batch = latent_codes_indices_batch.to(device)  
            latent_codes_batch = latent_codes_batch.to(device)

            predictions = self.model(x)  # (batch_size, 1)

            if self.train_cfg['clamp']:
                predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])

            loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape_full_exp(y, predictions, x[:, :self.train_cfg['latent_size']], sigma=self.train_cfg['sigma_regulariser'])
            loss_value.backward()       

            self.optimizer_latent.step()
            self.optimizer_model.step()
            total_loss += loss_value.data.cpu().numpy()  

            end_time = time.perf_counter()
            batch_time = end_time - start_time
            #tqdm.write(f"[Batch {batch_idx}] Time: {batch_time:.4f}s")
            
        avg_train_loss = total_loss/iterations
        print(f'Training: loss {avg_train_loss}')
        self.writer.add_scalar('Training loss', avg_train_loss, self.epoch)

        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        total_loss_rec = 0.0
        total_loss_latent = 0.0
        iterations = 0.0
        self.model.eval()

        for batch in val_loader:
            # batch[0]: [class, x, y, z], shape: (batch_size, 4)
            # batch[1]: [sdf], shape: (batch size)
            iterations += 1.0            

            x, y, _, latent_codes_batch = self.generate_xy(batch)

            x = x.to(device)  
            y = y.to(device)
            latent_codes_batch = latent_codes_batch.to(device)

            predictions = self.model(x)  # (batch_size, 1)
            if train_cfg['clamp']:
                predictions = torch.clamp(predictions, -train_cfg['clamp_value'], train_cfg['clamp_value'])

            loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape_full_exp(y, predictions, latent_codes_batch, self.train_cfg['sigma_regulariser'])          
            total_loss += loss_value.data.cpu().numpy()   
            total_loss_rec += loss_rec.data.cpu().numpy() 
            total_loss_latent += loss_latent.data.cpu().numpy()

        avg_val_loss = total_loss/iterations
        avg_loss_rec = total_loss_rec/iterations
        avg_loss_latent = total_loss_latent/iterations
        print(f'Validation: loss {avg_val_loss}')
        self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
        self.writer.add_scalar('Reconstruction loss', avg_loss_rec, self.epoch)
        self.writer.add_scalar('Latent code loss', avg_loss_latent, self.epoch)

        return avg_val_loss

if __name__=='__main__':
    train_cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), 'training.yaml')
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(train_cfg)
    trainer()   