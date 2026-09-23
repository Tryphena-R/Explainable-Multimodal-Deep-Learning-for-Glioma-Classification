import pandas as pd

def merge_radiogenomic_data(gene_path, mri_path, output_path):
    # 1. Load the datasets with correct formats
    print("Loading datasets...")
    gene_df = pd.read_csv(gene_path)  # Updated to read CSV
    mri_df = pd.read_excel(mri_path)  # Kept as Excel

    # 2. Resolve duplicate patient samples in the genomic dataset
    # This keeps only the first occurrence (primary tumor) and drops the rest
    initial_gene_count = len(gene_df)
    gene_df = gene_df.drop_duplicates(subset=['Patient ID'], keep='first')
    print(f"Dropped {initial_gene_count - len(gene_df)} duplicate genomic samples.")

    # 3. Perform the inner join (Early Fusion)
    # This ensures only patients with BOTH genomic and MRI data are retained
    fused_df = pd.merge(gene_df, mri_df, on='Patient ID', how='inner')
    
    # 4. Validate and save the final cohort
    print(f"Final fused dataset shape (Patients, Features): {fused_df.shape}")
    
    fused_df.to_csv(output_path, index=False)
    print(f"Successfully saved fused dataset to {output_path}")

# Execute the pipeline with your specific file extensions
merge_radiogenomic_data(
    gene_path='Gene_data.csv', 
    mri_path='MRI_Aggregated.xlsx', 
    output_path='Radiogenomic_Fused_Data.csv'
)
