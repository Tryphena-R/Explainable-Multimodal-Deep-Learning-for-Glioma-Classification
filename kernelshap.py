import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import shap
import matplotlib.pyplot as plt

df=pd.read_csv('Radiogenomic_Encoded.csv')
patient_col=[col for col in df.columns if 'Patient ID' in col][0]
df=df.drop_duplicates(subset=[patient_col],keep='first')
y=df['Tumor_Class'].values

leaky_keywords=['Patient ID','Tumor_Class','Collection','Project','Disease','Study','Unnamed']
drop_cols=[col for col in df.columns if any(k in col for k in leaky_keywords)]
X_df=df.drop(columns=drop_cols)
X_df=X_df.apply(pd.to_numeric,errors='coerce')
X_df=X_df.fillna(0)
X=X_df.values.astype(np.float32)

X_train_raw,X_test,y_train_raw,y_test=train_test_split(X,y,test_size=0.2,stratify=y)

smote=SMOTE()
X_train,y_train=smote.fit_resample(X_train_raw,y_train_raw)

X_train_t=torch.tensor(X_train,dtype=torch.float32)
y_train_t=torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
X_test_t=torch.tensor(X_test,dtype=torch.float32)
y_test_t=torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

train_ds=TensorDataset(X_train_t,y_train_t)
train_dl=DataLoader(train_ds,batch_size=32,shuffle=True)

class MLPClassifier(nn.Module):
    def __init__(self,input_dim):
        super(MLPClassifier,self).__init__()
        self.fc1=nn.Linear(input_dim,64)
        self.dropout=nn.Dropout(0.3)
        self.out=nn.Linear(64,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.out(x)
        res=self.sigmoid(x)
        return res

input_dim=X.shape[1]
model=MLPClassifier(input_dim)
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

best_loss=float('inf')
patience_counter=0

for epoch in range(100):
    model.train()
    total_loss=0.0
    for batch_X,batch_y in train_dl:
        optimizer.zero_grad()
        res_val=model(batch_X)
        loss=criterion(res_val,batch_y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(train_dl)
    print(f"Epoch:{epoch+1},Loss:{avg_loss:.4f}")
    if avg_loss<best_loss:
        best_loss=avg_loss
        patience_counter=0
    else:
        patience_counter+=1
    if patience_counter>=15:
        print("Early stopping triggered.")
        break

model.eval()
with torch.no_grad():
    test_res=model(X_test_t)
    res_preds=(test_res>0.5).float()
    acc=accuracy_score(y_test_t,res_preds)
    f1=f1_score(y_test_t,res_preds)
    auc=roc_auc_score(y_test_t,test_res)
    print(f"Accuracy:{acc:.4f},F1:{f1:.4f},AUC:{auc:.4f}")

def predict_fn(x):
    x_t=torch.tensor(x,dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        res_pred=model(x_t).numpy()
    return res_pred

background=X_train[:100]
explainer=shap.KernelExplainer(predict_fn,background)
test_samples=X_test[:50]
shap_values=explainer.shap_values(test_samples)

plt.clf()
feature_names=list(X_df.columns)
shap.summary_plot(shap_values,test_samples,feature_names=feature_names,plot_type="bar",max_display=10,show=False)
res=plt.gcf()
res.savefig('KernelSHAP_Global_Feature_Importance_Bar.png',bbox_inches='tight')
plt.show()
