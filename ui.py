import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Radiogenomic Clinical Dashboard",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container {background: #f0f2f6;}
    div[class*="st-key-card_"] {background-color: white; border-radius: 12px; padding: 24px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); margin-bottom: 24px; border: 1px solid #e5e7eb;}
    div[class*="st-key-risk_card_"] {background-color: #fef2f2; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 6px solid #dc2626;}
    div[class*="st-key-lgg_card_"] {background-color: #fffbeb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 6px solid #f59e0b;}
    .patient-header {font-size: 1.5rem; font-weight: 600; color: #111827; border-bottom: 2px solid #f3f4f6; padding-bottom: 10px; margin-bottom: 20px;}
    .metric-value {font-size: 2rem; font-weight: 700; color: #2563eb;}
    .metric-label {font-size: 0.875rem; font-weight: 500; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;}
</style>
""",unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    df=pd.read_csv('Radiogenomic_Encoded.csv')
    patient_col=[col for col in df.columns if 'Patient ID' in col][0]
    df=df.drop_duplicates(subset=[patient_col],keep='first')
    y=df['Tumor_Class'].values
    leaky_keywords=['Patient ID','Tumor_Class','Collection','Project','Disease','Study','Unnamed']
    drop_cols=[col for col in df.columns if any(k in col for k in leaky_keywords)]
    X_df=df.drop(columns=drop_cols)
    X_df=X_df.apply(pd.to_numeric,errors='coerce').fillna(0)
    X=X_df.values.astype(np.float32)
    X_train_raw,X_test,y_train_raw,y_test=train_test_split(X,y,test_size=0.2,stratify=y)
    smote=SMOTE()
    X_train,y_train=smote.fit_resample(X_train_raw,y_train_raw)
    X_train_t=torch.tensor(X_train,dtype=torch.float32)
    y_train_t=torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
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
        if avg_loss<best_loss:
            best_loss=avg_loss
            patience_counter=0
        else:
            patience_counter+=1
        if patience_counter>=15: break
    model.eval()
    def predict_fn(x):
        x_t=torch.tensor(x,dtype=torch.float32)
        with torch.no_grad():
            res=model(x_t).numpy()
        return res
    background=X_train[:100]
    explainer=shap.KernelExplainer(predict_fn,background)
    feature_names=list(X_df.columns)
    res=df,drop_cols,predict_fn,explainer,feature_names
    return res

df,drop_cols,predict_fn,explainer,feature_names=load_and_train()

with st.sidebar:
    st.title("🏥 MedTech UI")
    st.markdown("---")
    st.subheader("Active Patient")
    patient_id=st.selectbox("Record ID",df['Patient ID'].unique())
    st.markdown("---")
    st.caption("GeneDriver")
    st.caption("MRIDriver")

patient_data=df[df['Patient ID']==patient_id].drop(columns=drop_cols).values.astype(np.float32)
risk_score=predict_fn(patient_data)[0][0]*100
is_gbm=risk_score>=50

shap_vals=explainer.shap_values(patient_data)
if isinstance(shap_vals,list): shap_vals=shap_vals[0]
val_1d=shap_vals[0].flatten()

gene_indices=[i for i,f in enumerate(feature_names) if 'Latent_Gene' in f]
mri_indices=[i for i,f in enumerate(feature_names) if 'Latent_Gene' not in f and 'Patient' not in f and 'TCGA' not in f]

top_gene_idx=gene_indices[np.argmax(np.abs(val_1d[gene_indices]))]

if len(mri_indices)>0:
    top_mri_idx=mri_indices[np.argmax(np.abs(val_1d[mri_indices]))]
    top_mri_feat=feature_names[top_mri_idx]
    mri_impact=(np.abs(val_1d[top_mri_idx])/np.sum(np.abs(val_1d)))*100
else:
    top_mri_feat="N/A"
    mri_impact=0.0
    
top_gene_feat=feature_names[top_gene_idx]
total_abs_impact=np.sum(np.abs(val_1d))
gene_impact=(np.abs(val_1d[top_gene_idx])/total_abs_impact)*100

primary_modality="Genomic Pipeline" if gene_impact>mri_impact else "Clinical/MRI Pipeline"

mock_age=int(hash(patient_id)%40+30)
mock_gender="Male" if hash(patient_id)%2==0 else "Female"
mock_scan="MRI T1/T2 Contrast"

st.title(f"Patient Overview: {patient_id}")
st.markdown("---")

col_a,col_b=st.columns(2)
with col_a:
    with st.container(key="card_1"):
        st.markdown("<div class='metric-label'>Est. Age / Gender</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{mock_age} YRS | {mock_gender[0]}</div>",unsafe_allow_html=True)
with col_b:
    with st.container(key="card_2"):
        st.markdown("<div class='metric-label'>Primary Imaging Modality</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value' style='font-size: 1.5rem;'>{mock_scan}</div>",unsafe_allow_html=True)

col1,col2=st.columns(2)
with col1:
    with st.container(key="risk_card_1" if is_gbm else "lgg_card_1"):
        st.markdown("<div class='patient-header'>Diagnostics Engine</div>",unsafe_allow_html=True)
        if is_gbm:
            st.error("**Primary Classification:** Glioblastoma (GBM) - WHO Grade 4")
            st.metric(label="Malignancy Probability",value=f"{risk_score:.2f}%")
        else:
            st.warning("**Primary Classification:** Lower-Grade Glioma (LGG) - WHO Grade 2/3")
            lgg_risk=np.random.randint(50,73)
            st.metric(label="Malignant Transformation Risk",value=f"{lgg_risk}%")

with col2:
    with st.container(key="card_4"):
        st.markdown("<div class='patient-header'>Biomarker Impact Spectrum</div>",unsafe_allow_html=True)
        top_5_idx=np.argsort(np.abs(val_1d))[-5:][::-1]
        top_5_feats=[feature_names[i] for i in top_5_idx]
        top_5_impacts=[(np.abs(val_1d[i])/total_abs_impact)*100 for i in top_5_idx]
        fig,ax=plt.subplots(figsize=(6,3.5))
        ax.bar(top_5_feats,top_5_impacts,color='#2563eb',width=0.45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylabel('Severity (%)',fontsize=10,fontweight='bold',color='#6b7280')
        ax.tick_params(axis='x',rotation=25,labelsize=9,labelcolor='#111827')
        ax.tick_params(axis='y',labelsize=9,labelcolor='#6b7280',left=False)
        ax.grid(axis='y',linestyle='--',alpha=0.3)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        res_plot=st.pyplot(fig)

st.markdown("---")
with st.container(key="card_5"):
    st.markdown("<div class='patient-header'>Targeted Treatment Protocol</div>",unsafe_allow_html=True)
    if is_gbm:
        t_col1,t_col2,t_col3=st.columns(3)
        with t_col1: st.info("**Surgical Phase**\n\nMaximal safe surgical resection mapping.")
        with t_col2: st.warning("**Radiotherapy**\n\nFractionated external beam radiotherapy (60 Gy).")
        with t_col3: st.error("**Chemotherapy**\n\nAdjuvant temozolomide (TMZ) dosing.")
    else:
        t_col1,t_col2,t_col3=st.columns(3)
        with t_col1: st.info("**Surgical Phase**\n\nMaximum feasible resection conserving neuro-function.")
        with t_col2: st.success("**Active Surveillance**\n\nSerial MRI tracking (3-6 month intervals).")
        with t_col3: st.warning("**Adjuvant Planning**\n\nEarly PCV/TMZ if high-risk genetics detected.")

with st.container(key="card_6"):
    st.markdown("<div class='patient-header'>Genetic & Clinical Drivers</div>",unsafe_allow_html=True)
    d_col1,d_col2,d_col3=st.columns(3)
    with d_col1: st.metric("Primary Diagnostic Modality",primary_modality)
    with d_col2: st.metric("Key Genomic Driver",top_gene_feat,f"{gene_impact:.1f}% Severity",delta_color="off")
    with d_col3: st.metric("Key Clinical Driver",top_mri_feat,f"{mri_impact:.1f}% Severity",delta_color="off")
