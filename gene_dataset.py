import pandas as pd

# Pandas unzips and reads the file in memory automatically
gene_df = pd.read_csv('TCGA.GBMLGG.sampleMap_HiSeqV2_PANCAN.gz', sep='\t', compression='gzip')

print(f"Genomic dataset shape: {gene_df.shape}")
