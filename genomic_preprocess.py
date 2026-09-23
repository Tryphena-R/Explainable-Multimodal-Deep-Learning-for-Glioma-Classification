import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def preprocess_genomic_data(expression_matrix):
    """
    Applies log2 scaling and Z-score normalization to gene expression data.
    """
    # 1. log2 Scaling (adding a small constant to prevent log(0) errors)
    log_scaled_data = np.log2(expression_matrix + 1e-9)
    
    # 2. Z-Score Normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(log_scaled_data)
    
    # Convert to PyTorch tensors for the neural network
    return torch.tensor(normalized_data, dtype=torch.float32)

# Example usage (assuming 'gene_features' is your subset matrix):
# processed_genomic_tensor = preprocess_genomic_data(gene_features)
