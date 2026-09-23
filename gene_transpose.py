import pandas as pd

gene_df = pd.read_csv('TCGA.GBMLGG.sampleMap_HiSeqV2_PANCAN.gz', sep='\t', index_col=0, compression='gzip')
gene_df = gene_df.T
gene_df.reset_index(inplace=True)
gene_df.rename(columns={'index': 'Patient ID'}, inplace=True)

gene_df['Patient ID'] = gene_df['Patient ID'].astype(str).str[:12]
gene_df = gene_df.drop_duplicates(subset=['Patient ID'], keep='first')

gbm_mri = pd.read_excel('TCIA_TCGA-GBM_09-16-2015-nbia-digest.xlsx')
lgg_mri = pd.read_excel('TCIA_TCGA-LGG_09-16-2015-nbia-digest.xlsx')

gbm_mri['Tumor_Class'] = 1
lgg_mri['Tumor_Class'] = 0
all_mri = pd.concat([gbm_mri, lgg_mri], ignore_index=True)

fused_df = pd.merge(gene_df, all_mri, on='Patient ID', how='inner')

fused_df.to_csv('Radiogenomic_Fused_Data_Complete.csv', index=False)
print(fused_df.shape)
print(fused_df['Tumor_Class'].value_counts())
