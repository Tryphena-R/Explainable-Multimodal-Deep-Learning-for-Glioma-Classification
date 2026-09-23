import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

class GenomicAutoencoder(nn.Module):
    def __init__(self,input_dim):
        super(GenomicAutoencoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )
        self.decoder=nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,input_dim)
        )
    def forward(self,x):
        z=self.encoder(x)
        res=self.decoder(z)
        return res,z
