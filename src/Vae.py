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
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # 128x128
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),  # 64x64
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), # 32x32
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1), # 16x16
            nn.ReLU(),

            nn.Conv2d(256, 512, 4, 2, 1), # 8x8
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

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

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 128x128
            nn.ReLU(),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1), # 256x256
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 8, 8)
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


def loss_function(x_hat, x, mu, logvar, beta=1):
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


def train_vae(vae_model, X_train, X_val, epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-5, beta=1, device=None):
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
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'train_kld': [],
        'val_recon_loss': [],
        'val_kld': []
    }
    best_val_loss = float('inf')
    best_model_state = None

    print(f"Starting VAE training on {device}...")
    
    for epoch in range(epochs):
        vae_model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld = 0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae_model(data)
            loss, recon_loss, kld = loss_function(x_hat, data, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kld += kld.item()
            if batch_idx % 10 == 0:
                pass

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_recon_loss = train_recon_loss / len(train_loader.dataset)
        avg_train_kld = train_kld / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        history['train_recon_loss'].append(avg_train_recon_loss)
        history['train_kld'].append(avg_train_kld)

        # Validation
        vae_model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld = 0
        with torch.no_grad():
            for data, in val_loader:
                data = data.to(device)
                x_hat, mu, logvar = vae_model(data)
                loss, recon_loss, kld = loss_function(x_hat, data, mu, logvar, beta=beta)
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kld += kld.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_recon_loss = val_recon_loss / len(val_loader.dataset)
        avg_val_kld = val_kld / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)
        history['val_recon_loss'].append(avg_val_recon_loss)
        history['val_kld'].append(avg_val_kld)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Recon: {avg_train_recon_loss:.4f}, Train KL: {avg_train_kld:.4f}, "
              f"Val Recon: {avg_val_recon_loss:.4f}, Val KL: {avg_val_kld:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(vae_model.state_dict())

    print("VAE training complete!")

    if best_model_state is not None:
        print(f"Restoring best model from epoch with val loss: {best_val_loss:.4f}")
        vae_model.load_state_dict(best_model_state)

    return history


