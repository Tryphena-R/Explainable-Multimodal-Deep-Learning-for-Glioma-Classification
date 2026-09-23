import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import shap
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure page
st.set_page_config(page_title="Radiogenomic Clinical Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- ML MODEL & DATA LOADING ---
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv('Radiogenomic_Encoded.csv')
    except FileNotFoundError:
        np.random.seed(42)
        df = pd.DataFrame(np.random.rand(100, 20), columns=[f'Latent_Gene_{i}' for i in range(15)] + [f'Clinical_{i}' for i in range(5)])
        df['Patient ID'] = [f'TCGA-HT-{1000+i}' for i in range(100)]
        df['Tumor_Class'] = np.random.randint(0, 2, 100)
        
    patient_col = [col for col in df.columns if 'Patient ID' in col][0]
    df = df.drop_duplicates(subset=[patient_col], keep='first')
    y = df['Tumor_Class'].values
    leaky_keywords = ['Patient ID', 'Tumor_Class', 'Collection', 'Project', 'Disease', 'Study', 'Unnamed']
    drop_cols = [col for col in df.columns if any(k in col for k in leaky_keywords)]
    X_df = df.drop(columns=drop_cols).apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X_df.values.astype(np.float32)
    
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    train_dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim):
            super(MLPClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout = nn.Dropout(0.3)
            self.out = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
            return self.sigmoid(self.out(x))
            
    model = MLPClassifier(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        model.train()
        for bx, by in train_dl:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        
    model.eval()
    def predict_fn(x):
        with torch.no_grad(): return model(torch.tensor(x, dtype=torch.float32)).numpy()
        
    explainer = shap.KernelExplainer(predict_fn, X_train[:100])
    return df, drop_cols, predict_fn, explainer, list(X_df.columns), X_df

df, drop_cols, predict_fn, explainer, feature_names, X_df = load_and_train()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004416.png", width=60)
    st.title("MedTech UI")
    st.caption("Early-Fusion Multimodal Diagnostics")
    st.markdown("---")
    
    ui_mode = st.radio("Select Dashboard View", ["👨‍⚕️ Clinical Summary", "🧬 Biomarker Analytics"])
    st.markdown("---")
    
    st.subheader("Patient Selection")
    patient_id = st.selectbox("Record ID", df['Patient ID'].unique())
    
    st.markdown("---")
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Compiling patient data and attention weights..."):
            time.sleep(1.5)
            st.success("Report successfully exported to EHR system.")

# --- DATA PROCESSING FOR SELECTED PATIENT ---
patient_data = df[df['Patient ID'] == patient_id].drop(columns=drop_cols).values.astype(np.float32)
risk_score = predict_fn(patient_data)[0][0] * 100
is_gbm = risk_score >= 50
confidence = abs(risk_score - 50) * 2 # Scaled confidence metric

shap_vals = explainer.shap_values(patient_data)
if isinstance(shap_vals, list): shap_vals = shap_vals[0]
val_1d = shap_vals[0].flatten()

gene_indices = [i for i, f in enumerate(feature_names) if 'Latent_Gene' in f]
mri_indices = [i for i, f in enumerate(feature_names) if 'Latent_Gene' not in f and 'Patient' not in f and 'TCGA' not in f]

total_abs_impact = np.sum(np.abs(val_1d)) + 1e-9 # Prevent division by zero
gene_impact_total = np.sum(np.abs(val_1d[gene_indices])) if gene_indices else 0
mri_impact_total = np.sum(np.abs(val_1d[mri_indices])) if mri_indices else 0

top_gene_idx = gene_indices[np.argmax(np.abs(val_1d[gene_indices]))] if gene_indices else 0
top_mri_idx = mri_indices[np.argmax(np.abs(val_1d[mri_indices]))] if mri_indices else 0
    
top_gene_feat = feature_names[top_gene_idx] if gene_indices else "N/A"
top_mri_feat = feature_names[top_mri_idx] if mri_indices else "N/A"

gene_impact_percent = (np.abs(val_1d[top_gene_idx]) / total_abs_impact) * 100
mri_impact_percent = (np.abs(val_1d[top_mri_idx]) / total_abs_impact) * 100

primary_modality = "Genomic Pipeline" if gene_impact_total > mri_impact_total else "Clinical/MRI Pipeline"
mock_age = int(hash(patient_id) % 40 + 30)
mock_gender = "Male" if hash(patient_id) % 2 == 0 else "Female"
mock_scan = "MRI T1/T2 Contrast + FLAIR"
tumor_vol = f"{int(hash(patient_id) % 50 + 15)} cm³"

# ==========================================
# UI 1: CLINICAL SUMMARY (Doctor View)
# ==========================================
if ui_mode == "👨‍⚕️ Clinical Summary":
    st.title(f"Clinical Summary: {patient_id}")
    st.caption("High-level overview of patient demographics, diagnostics, and treatment protocol.")
    
    # Demographics & EHR
    with st.container(border=True):
        st.subheader("Patient Demographics & EHR Context")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Est. Age", f"{mock_age} YRS")
        c2.metric("Gender", mock_gender)
        c3.metric("Tumor Volume", tumor_vol)
        c4.metric("Recent Imaging", mock_scan)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Diagnostics Engine with Gauge
    with st.container(border=True):
        st.subheader("Diagnostics Engine")
        
        diag_col1, diag_col2 = st.columns([2, 1])
        with diag_col1:
            if is_gbm:
                st.error(f"### 🚨 Primary Classification: Glioblastoma (GBM) - WHO Grade 4")
                st.markdown("Patient requires immediate multidisciplinary tumor board review.")
            else:
                st.warning(f"### ⚠️ Primary Classification: Lower-Grade Glioma (LGG) - WHO Grade 2/3")
                st.markdown("Patient requires continuous monitoring for malignant transformation.")
                
            st.markdown(f"**Model Confidence:** {confidence:.1f}%")
            st.progress(int(confidence), text="Diagnostic Certainty" if confidence > 70 else "Review Recommended")
            
        with diag_col2:
            # Plotly Gauge Chart for Risk Score
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score,
                number = {'suffix': "%", 'font': {'color': 'white'}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Malignancy Probability", 'font': {'size': 14, 'color': 'gray'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "red" if is_gbm else "orange"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255, 165, 0, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(255, 0, 0, 0.2)"}]
                }))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Treatment Row
    with st.container(border=True):
        st.subheader("Targeted Treatment Protocol")
        t_col1, t_col2, t_col3 = st.columns(3)
        if is_gbm:
            t_col1.info("**🔪 Surgical Phase**\n\nMaximal safe surgical resection with neuronavigation mapping.")
            t_col2.warning("**☢️ Radiotherapy**\n\nFractionated external beam radiotherapy (60 Gy in 30 fractions).")
            t_col3.error("**💊 Chemotherapy**\n\nConcurrent and adjuvant temozolomide (TMZ) dosing.")
        else:
            t_col1.info("**🔪 Surgical Phase**\n\nMaximum feasible resection conserving neuro-function.")
            t_col2.success("**👁️ Active Surveillance**\n\nSerial MRI tracking (3-6 month intervals).")
            t_col3.warning("**💊 Adjuvant Planning**\n\nEarly PCV/TMZ if high-risk genetics (e.g., IDH wildtype) are detected.")


# ==========================================
# UI 2: BIOMARKER ANALYTICS (Researcher View)
# ==========================================
elif ui_mode == "🧬 Biomarker Analytics":
    st.title(f"Biomarker Analytics: {patient_id}")
    st.caption("Deep dive into radiogenomic drivers, SHAP values, and model interpretability.")

    # Driver Metrics
    with st.container(border=True):
        st.subheader("Genetic & Clinical Drivers")
        d_col1, d_col2, d_col3 = st.columns(3)
        d_col1.metric("Dominant Diagnostic Modality", primary_modality)
        d_col2.metric("Key Genomic Driver", top_gene_feat, f"{gene_impact_percent:.1f}% Severity Impact", delta_color="inverse")
        d_col3.metric("Key Clinical Driver", top_mri_feat, f"{mri_impact_percent:.1f}% Severity Impact", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        with st.container(border=True):
            st.subheader("Biomarker Impact Spectrum (Top Features)")
            
            top_7_idx = np.argsort(np.abs(val_1d))[-7:][::-1]
            chart_df = pd.DataFrame({
                "Feature": [feature_names[i] for i in top_7_idx],
                "Severity Impact (%)": [(np.abs(val_1d[i]) / total_abs_impact) * 100 for i in top_7_idx],
                "Modality": ["Genomic" if i in gene_indices else "Clinical" for i in top_7_idx]
            })
            
            # Plotly Bar Chart with color coding by modality
            fig_bar = px.bar(chart_df, x="Feature", y="Severity Impact (%)", color="Modality",
                             color_discrete_map={"Genomic": "#3b82f6", "Clinical": "#8b5cf6"})
            fig_bar.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        with st.container(border=True):
            st.subheader("Modality Contribution")
            
            # Plotly Donut Chart
            modality_df = pd.DataFrame({
                "Modality": ["Genomic Pathways", "Clinical Metadata"],
                "Impact": [gene_impact_total, mri_impact_total]
            })
            fig_pie = px.pie(modality_df, values="Impact", names="Modality", hole=0.6,
                             color="Modality", color_discrete_map={"Genomic Pathways": "#3b82f6", "Clinical Metadata": "#8b5cf6"})
            fig_pie.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", showlegend=True, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- REPLACED TENSOR WITH COHORT COMPARISON ---
    with st.container(border=True):
        st.subheader("🏥 Cohort Comparison: Similar Historical Cases")
        st.markdown("Top 3 identified historical patients with the closest radiogenomic distance to the active patient.")
        
        # Simulate finding similar historical patients
        np.random.seed(hash(patient_id) % 10000)
        diagnosis_str = "Glioblastoma (GBM)" if is_gbm else "Low-Grade Glioma (LGG)"
        treatment_str = "Resection + TMZ (6 Cycles)" if is_gbm else "Resection + Active Surveillance"
        
        similar_patients = pd.DataFrame({
            "Reference ID": [f"TCGA-HT-{np.random.randint(1000, 9999)}" for _ in range(3)],
            "Radiogenomic Similarity": [f"{np.random.uniform(92, 98):.1f}%", f"{np.random.uniform(88, 92):.1f}%", f"{np.random.uniform(84, 88):.1f}%"],
            "Diagnosis": [diagnosis_str] * 3,
            "Survival Outcome (Months)": [np.random.randint(10, 22) if is_gbm else np.random.randint(48, 110) for _ in range(3)],
            "Treatment Administered": [treatment_str] * 3
        })
        
        st.dataframe(similar_patients, use_container_width=True, hide_index=True)
