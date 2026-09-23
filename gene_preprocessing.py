import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

# Suppress standard mathematical warnings for a cleaner console output
warnings.filterwarnings('ignore', category=RuntimeWarning)

print("Loading fused dataset...")
df = pd.read_csv('Radiogenomic_Fused_Data_Complete.csv', low_memory=False)

patient_ids = df['Patient ID']
y = df['Tumor_Class']

# -----------------------------------------
# 1. GENOMIC PIPELINE
# -----------------------------------------
print("Processing Genomic Data...")
gene_cols = [col for col in df.columns if col not in ['Patient ID', 'Tumor_Class'] and not col.startswith(('num_', 'total_', 'Patient ', 'Study '))]
genes_df = df[gene_cols].copy()

genes_df = genes_df.apply(pd.to_numeric, errors='coerce')
genes_df = genes_df.select_dtypes(include=[np.number])

# Safeguard 1: Drop genes that are 100% empty, then fill remaining NaNs with 0
genes_df.dropna(axis=1, how='all', inplace=True)
genes_df.fillna(0, inplace=True)

# log2 scaling
genes_df = np.log2(genes_df - genes_df.min().min() + 1)

# Z-score normalization
scaler_z = StandardScaler()
genes_scaled = pd.DataFrame(scaler_z.fit_transform(genes_df), columns=genes_df.columns)

# -----------------------------------------
# 2. CLINICAL PIPELINE
# -----------------------------------------
print("Processing Clinical Metadata...")
clinical_cols = [col for col in df.columns if col not in gene_cols and col not in ['Patient ID', 'Tumor_Class']]
clinical_df = df[clinical_cols].copy()

# Safeguard 2: Drop completely empty clinical columns (like 'Study ID') before imputation
clinical_df.dropna(axis=1, how='all', inplace=True)

cat_cols = clinical_df.select_dtypes(include=['object', 'bool']).columns
cont_cols = clinical_df.select_dtypes(exclude=['object', 'bool']).columns

# Median Imputation & Min-Max Scaling
imputer_median = SimpleImputer(strategy='median')
if len(cont_cols) > 0:
    clinical_df[cont_cols] = imputer_median.fit_transform(clinical_df[cont_cols])
    
    scaler_minmax = MinMaxScaler()
    # Ensure it writes back as a DataFrame to keep the columns aligned
    clinical_df[cont_cols] = pd.DataFrame(
        scaler_minmax.fit_transform(clinical_df[cont_cols]), 
        columns=cont_cols, 
        index=clinical_df.index
    )

clinical_encoded = pd.get_dummies(clinical_df, columns=cat_cols, drop_first=True)

# -----------------------------------------
# 3. EARLY FUSION
# -----------------------------------------
print("Fusing preprocessed tensors...")
processed_df = pd.concat([patient_ids, genes_scaled, clinical_encoded, y], axis=1)

processed_df.to_csv('Radiogenomic_Preprocessed_Ready.csv', index=False)
print(f"Preprocessed dataset saved. Final shape: {processed_df.shape}")
