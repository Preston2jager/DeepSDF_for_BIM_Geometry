import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.train_cfg = config
        in_channels = config['GCN_feature_channels'] + config['GCN_tag_dim']
        out_channels = config['GCN_feature_channels']
        hidden_channels = config['GCN_hidden_channels']
        self.dropout = config.get('dropout', 0.5)
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.conv3]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, g, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(g, x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(g, x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(g, x)

        return x  # (num_nodes, out_dim)

