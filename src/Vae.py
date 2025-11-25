import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Encoder (Updated with BatchNorm + LeakyReLU)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        super().__init__()
        
        # Helper to create layers quickly
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, 2, 1), # Stride 2 downsampling
                nn.BatchNorm2d(out_f),           # Batch Norm for stability
                nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU for better gradients
            )

        self.c1 = conv_block(img_channels, 32)   # 64x64
        self.c2 = conv_block(32, 64)             # 32x32
        self.c3 = conv_block(64, 128)            # 16x16
        self.c4 = conv_block(128, 256)           # 8x8

        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = h.flatten(1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# -----------------------------
# Decoder (Updated with BatchNorm + LeakyReLU)
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        def deconv_block(in_f, out_f):
            return nn.Sequential(
                nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.d1 = deconv_block(256, 128) # 16x16
        self.d2 = deconv_block(128, 64)  # 32x32
        self.d3 = deconv_block(64, 32)   # 64x64
        
        # Final layer no BN, Tanh activation for image
        self.d4 = nn.ConvTranspose2d(32, img_channels, 4, 2, 1) # 128x128

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 8, 8)
        
        h = self.d1(h)
        h = self.d2(h)
        h = self.d3(h)
        # Output layer
        return torch.tanh(self.d4(h)) 

# -----------------------------
# VAE Wrapper
# -----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        super().__init__()
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
# IMPROVED LOSS FUNCTION
# -----------------------------
def loss_function(x_hat, x, mu, logvar, beta=0.5):
    # Switch from MSE to L1 (MAE) for sharper images
    recon_loss = F.l1_loss(x_hat, x, reduction="sum")
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # We lower beta slightly to prioritize reconstruction (Image Quality) over Latent Structure
    return recon_loss + (beta * kld), recon_loss, kld

# -----------------------------
# TRAINING LOOP (Minor adjustments)
# -----------------------------
def train_vae(vae_model, X_train, X_val, epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-4, beta=0.2, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae_model.to(device)
    
    # Ensure inputs are tensors
    if not isinstance(X_train, torch.Tensor):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    else:
        X_train_tensor = X_train
        X_val_tensor = X_val

    # If data is (N, 128, 128), add channel dim -> (N, 1, 128, 128)
    if X_train_tensor.ndim == 3:
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_val_tensor = X_val_tensor.unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(vae_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Starting VAE training on {device}...")
    
    for epoch in range(epochs):
        vae_model.train()
        train_loss = 0
        
        for (data,) in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            x_hat, mu, logvar = vae_model(data)
            
            # Use the new Loss function
            loss, recon, kld = loss_function(x_hat, data, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        vae_model.eval()
        val_loss = 0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(device)
                x_hat, mu, logvar = vae_model(data)
                loss, recon, kld = loss_function(x_hat, data, mu, logvar, beta=beta)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return history