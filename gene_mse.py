import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

df=pd.read_csv('Radiogenomic_Preprocessed_Ready.csv',low_memory=False)
gene_cols=[col for col in df.columns if col not in ['Patient ID','Tumor_Class'] and not col.startswith(('num_','total_','Patient ','Study '))]
genes_data=df[gene_cols].values
tensor_x=torch.tensor(genes_data,dtype=torch.float32)
dataset=TensorDataset(tensor_x,tensor_x)
dataloader=DataLoader(dataset,batch_size=32,shuffle=True)
input_dim=len(gene_cols)

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

model=GenomicAutoencoder(input_dim)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

for epoch in range(50):
    total_loss=0.0
    for data in dataloader:
        inputs,_=data
        optimizer.zero_grad()
        res,z=model(inputs)
        loss=criterion(res,inputs)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch:{epoch+1},Loss:{total_loss/len(dataloader)}")

_,latent_vectors=model(tensor_x)
latent_df=pd.DataFrame(latent_vectors.detach().numpy())
latent_df.columns=[f"Latent_Gene_{i}" for i in range(128)]
res=pd.concat([df['Patient ID'],latent_df,df.drop(columns=gene_cols)],axis=1)
res.to_csv('Radiogenomic_Encoded.csv',index=False)
