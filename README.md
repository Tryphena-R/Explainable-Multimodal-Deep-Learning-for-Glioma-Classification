Explainable Multimodal Deep Learning for Glioma Classification
📌 Overview

This project presents an explainable multimodal deep learning framework for classifying brain tumors into Glioblastoma (GBM) and Low-Grade Glioma (LGG).

The framework integrates two complementary data sources:

🧬 Gene Expression Data – High-dimensional transcriptomic features from TCGA.
🧠 MRI Acquisition Metadata – Clinical and scan-level parameters from TCIA.

The combined information is processed using an Autoencoder, Multilayer Perceptron (MLP), and KernelSHAP to achieve accurate and interpretable tumor classification.

🏗️ System Architecture
                Gene Expression Data
                        │
              Log2 Transformation
                        │
                Z-Score Scaling
                        │
                  Autoencoder
                        │
                  128-D Latent
                   Representation
                        │
                        ├──────────────┐
                        │              │
                        │        Feature Fusion
                        │              │
MRI Acquisition Metadata              │
        │                              │
Median Imputation                      │
        │                              │
One-Hot Encoding                       │
        │                              │
Min-Max Scaling                        │
        │                              │
Clinical Feature Vector ───────────────┘
                       │
                Regularized MLP
                       │
              ┌────────┴────────┐
              │                 │
             GBM               LGG
              │                 │
              └────────┬────────┘
                       │
                  KernelSHAP
                       │
             Feature Explanations

The proposed architecture performs early feature fusion before classification and applies KernelSHAP as a post-hoc explainability module.

🔬 Methodology
1. Gene Expression Processing

Gene expression data containing more than 20,000 features is:

Log2 transformed
Z-score standardized
Processed using an undercomplete Autoencoder
Compressed into a 128-dimensional latent representation

The 128-dimensional bottleneck was selected after evaluating multiple latent dimensions.

2. MRI Metadata Processing

MRI acquisition and clinical parameters include features such as:

Age
Gender
Slice Thickness
Number of Slices
TR and TE
Flip Angle
Field Strength
MRI Sequence Information
Tumor Volume
Contrast Enhancement Status

Missing numerical values are handled using median imputation, categorical variables are one-hot encoded, and continuous features are Min-Max scaled.

3. Multimodal Feature Fusion

The 128-dimensional genomic representation is concatenated with the MRI/clinical feature vector to create a unified multimodal representation.

Fused Representation = [Gene Latent Features || MRI Metadata]

An MLP classifier then learns relationships between the two modalities and predicts GBM or LGG.

4. Explainable AI

KernelSHAP is used to explain individual model predictions by estimating the contribution of each feature.

This provides both:

Patient-level feature attribution
Global feature importance

The analysis identifies features such as contrast enhancement, tumor volume, and latent gene representations among the influential features.

📊 Dataset
Parameter	Value
Total Patients	1,250
GBM Cases	600
LGG Cases	650
Gene Expression Features	>20,000
MRI Metadata Features	15
Autoencoder Latent Dimension	128
Training Samples	1,000
Testing Samples	250

🤖 Model Configuration
Component	Configuration
Autoencoder Epochs	50
MLP Epochs	100
Batch Size	32
Learning Rate	0.001
Optimizer	Adam
Autoencoder Loss	MSE
MLP Loss	Binary Cross-Entropy
MLP Dropout	0.3
Early Stopping	15 epochs
Explainability	KernelSHAP

📈 Results

The proposed multimodal DNN achieved:

Metric	Score
Accuracy	95.20%
Precision	0.948
Recall	0.941
F1-Score	0.944
AUC	0.971

The paper also reports an ablation accuracy of 74.1% using MRI metadata alone, 88.2% using gene expression alone, and 95.2% using early fusion.

🧠 Explainability

KernelSHAP provides an interpretable view of the model's decisions.

The global feature importance analysis highlights:

Contrast Status
Latent Gene 42
Tumour Volume
Latent Gene 112
Patient Age
Latent Gene 7

These feature contributions help provide insight into the factors influencing GBM/LGG predictions.

🛠️ Technologies
Python
Deep Learning
Autoencoders
Multilayer Perceptron (MLP)
KernelSHAP
Machine Learning
Explainable AI (XAI)
Multimodal Learning
Radiogenomics
TCGA
TCIA
🚀 Future Work

Future improvements include:

Integration of raw 3D MRI volumes
3D CNN and Vision Transformer architectures
Dynamic cross-attention between modalities
Handling missing modalities
Multi-grade glioma classification
Survival prediction
Larger multi-center clinical validation
Edge/low-latency deployment
