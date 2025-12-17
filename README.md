# Investigating Variational Autoencoders for Data Imbalance Mitigation in Solar Panel Defect Detection

This project focuses on defect detection in solar panels using machine learning models. It utilizes the ELPV (Electroluminescence Photovoltaic) dataset for training and evaluation. The project includes Variational Autoencoders (VAEs) for synthetic data generation to augment the dataset and a ResNet18 model for binary classification of defects.

## Folder Structure

- `src/`: Source code directory
  - `pre_processing.py`: Handles data loading from the ELPV dataset, image resizing, normalization, data augmentation (flips and rotations), and splitting into train/validation/test sets.
  - `plots.py`: Contains functions for plotting data distributions and visualizing data augmentations.
  - `VaeA.py`: Implementation of Variational Autoencoder A, used for generating synthetic defect samples to balance the dataset.
  - `Resnet18.py`: ResNet18 model adapted for binary classification on grayscale images.
  - `other_Vae_options/`: Directory containing alternative VAE implementations (VaeB, VaeC, VaeD) for experimentation.
- `Notebook.ipynb`: Jupyter notebook that demonstrates the entire workflow: data preprocessing, augmentation, loading pre-trained VAE models, generating synthetic samples, ResNet18 Performances
- `pyproject.toml`: Project configuration file defining dependencies and build settings.
- `Final_report.pdf`: The 2 pages lenght Report

## Model Weights

model weights (VAE, Resnet18) can be downloaded from: <[https://drive.google.com/drive/folders1rCzs0FLY0nW0CQRnHPsKPAmqpXf9Uzdv?usp=drive_link](https://drive.google.com/drive/folders/1rCzs0FLY0nW0CQRnHPsKPAmqpXf9Uzdv?usp=sharing)>


