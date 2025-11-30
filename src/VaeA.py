# implementation 
# 128x128 grayscale images
# normalized to [-1, 1]
# MSE or L1 loss for reconstruction

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 64x64
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32x32
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), # 16x16
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1), # 8x8
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1), # 128x128
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 8, 8)
        return self.deconv(h)


# -----------------------------
# VAE (Encoder + Decoder)
# -----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, img_channels)
        self.decoder = Decoder(latent_dim, img_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


# -----------------------------
# Loss Function
# -----------------------------
def loss_function(x_hat, x, mu, logvar, beta=1, recon="mse"):
    # Reconstruction loss
    if recon.lower() == "l1":
        recon_loss = F.l1_loss(x_hat, x, reduction="sum")

    elif recon.lower() == "mse":
        recon_loss = F.mse_loss(x_hat, x, reduction="sum")

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld * beta


def train_vae(vae_model, X_train, X_val, epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-4, beta=1, recon='mse', device=None):
    """
    Trains the VAE model.
    
    Args:
        vae_model: The VAE model instance.
        X_train: Training data (numpy array).
        X_val: Validation data (numpy array).
        epochs: Number of epochs to train.
        batch_size: Batch size for DataLoader.
        lr: Learning rate for Adam optimizer.
        device: 'cuda' or 'cpu'. If None, automatically detects.
        
    Returns:
        history: Dictionary containing 'train_loss' and 'val_loss' lists.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae_model.to(device)
    
    if X_train.ndim == 3:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    # Create TensorDatasets---------------------
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)

    # Create DataLoaders------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer---------------------------------
    optimizer = optim.Adam(vae_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training Loop-----------------------------
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None

    print(f"Starting VAE training on {device}...")
    
    for epoch in range(epochs):
        vae_model.train()
        train_loss = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            x_hat, mu, logvar = vae_model(data)
            loss = loss_function(x_hat, data, mu, logvar, beta=beta, recon=recon)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                pass

        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # Validation
        vae_model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, in val_loader:
                data = data.to(device)
                x_hat, mu, logvar = vae_model(data)
                loss = loss_function(x_hat, data, mu, logvar, beta=beta, recon=recon)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(vae_model.state_dict())
            # print(f"New best model found at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    print("VAE training complete!")
    
    if best_model_state is not None:
        print(f"Restoring best model from epoch with val loss: {best_val_loss:.4f}")
        vae_model.load_state_dict(best_model_state)
        
    return history


def generate_sample(vae_model, n_samples=800, device="cpu"):

    """
    Generates new samples from the VAE model by sampling from the latent space.
    
    Args:
        vae_model: Trained VAE instance.
        n_samples: How many new images to generate.
        device: 'cuda' or 'cpu'.
        
    Returns:
        generated_images: Numpy array of generated images.
    """
    # set a random seed for reproducibility
    torch.manual_seed(1927)

    vae_model.eval()
    vae_model.to(device)
    
    latent_dim = vae_model.latent_dim
    n_samples = n_samples

    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        generated = vae_model.decoder(z)
    
    return generated.squeeze().cpu().numpy()


def slerp_torch(val, low, high):
    """
    Spherical Linear Interpolation for PyTorch Tensors.
    """
    # 1. Normalize vectors to unit sphere to calculate the angle
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    
    # 2. Calculate the dot product (cosine of the angle)
    dot = (low_norm * high_norm).sum(1)
    
    # 3. Clamp for numerical stability (to ensure it stays between -1 and 1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # 4. Calculate the angle (omega)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    # 5. Handle case where vectors are parallel (omega = 0) to avoid div by zero
    # We treat 'so' as a scalar here for the check
    if so.item() < 1e-6:
        return (1.0 - val) * low + val * high
    
    # 6. Calculate interpolation coefficients
    s0 = torch.sin((1.0 - val) * omega) / so
    s1 = torch.sin(val * omega) / so
    
    # 7. Apply to ORIGINAL vectors (to preserve the latent radius/magnitude)
    return s0.unsqueeze(1) * low + s1.unsqueeze(1) * high


def generate_controlled_samples(vae_model, X_train, Z_train, X_healthy, n_samples=800, mode="spherical", threshold=5.0, device="cpu"):
    """
    Generates samples, filtering out those close to the 'Healthy' distribution.
    Optimized for ~1500 healthy reference images (no batching).
    """
    torch.manual_seed(1927)
    np.random.seed(1927)

    vae_model.eval()
    vae_model.to(device)
    
    generated_images = []
    
    # Convert Defective Training Data
    if not isinstance(X_train, torch.Tensor):
        X_data = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_data = X_train

    # Convert Healthy Reference Data
    if not isinstance(X_healthy, torch.Tensor):
        X_healthy_tensor = torch.tensor(X_healthy, dtype=torch.float32)
    else:
        X_healthy_tensor = X_healthy
        
    # Ensure correct shape (N, C, H, W) -> Adds channel dim if missing
    if len(X_healthy_tensor.shape) == 3:
        X_healthy_tensor = X_healthy_tensor.unsqueeze(1)

    # CALCULATE HEALTHY CENTROID---
    print("Calculating Healthy Centroid...")
    
    with torch.no_grad():

        mu_healthy, _ = vae_model.encoder(X_healthy_tensor.to(device))
        healthy_centroid = torch.mean(mu_healthy, dim=0)
    
    print(f"Centroid calculated. Filtering samples with Latent Distance < {threshold}")

    #GENERATION 
    count = 0
    attempts = 0
    max_attempts = n_samples * 10 # Allow more attempts to find good samples

    with torch.no_grad():
        while count < n_samples:
            attempts += 1
            if attempts > max_attempts:
                print("Warning: Max attempts reached. Stopped early.")
                break

            # A. Select class
            classes = ['poly', 'mono']
            s = np.random.choice(classes)
            indices = np.where(Z_train == s)[0]

            if len(indices) < 2: continue
                
            idx1, idx2 = np.random.choice(indices, size=2, replace=False)

            img1 = X_data[idx1].unsqueeze(0).unsqueeze(0).to(device) 
            img2 = X_data[idx2].unsqueeze(0).unsqueeze(0).to(device)

            mu1, _ = vae_model.encoder(img1)
            mu2, _ = vae_model.encoder(img2)
                
            alpha = np.random.uniform(0.2, 0.8) 
            
            # B. Interpolate
            if mode == "linear":
                z_new = (alpha * mu1) + ((1 - alpha) * mu2)
            elif mode == "spherical":
                z_new = slerp_torch(alpha, mu1, mu2)
            else:
                raise ValueError("Mode must be 'linear' or 'spherical'")

            
            # Calculate Euclidean distance
            distance = torch.norm(z_new - healthy_centroid)
            
            if distance.item() < threshold:
                # Too close to healthy -> Skip
                continue 
            
            # C. Decode
            x_new = vae_model.decoder(z_new)
            generated_images.append(x_new.squeeze().cpu().numpy())
            count += 1
            
            if count % 100 == 0:
                print(f"Generated {count}/{n_samples} images...")

    return np.array(generated_images)
