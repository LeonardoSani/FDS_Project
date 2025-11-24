import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    """
    Standard ResBlock: x + Conv -> BN -> ReLU -> Conv -> BN
    Keeps dimensions constant.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim

        # Helper: Downsample (stride 2) -> BN -> ReLU -> ResidualBlock
        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False), 
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                ResidualBlock(out_c)
            )

        self.net = nn.Sequential(
            # Input: 1 x 256 x 256
            down_block(img_channels, 32),  # -> 128x128
            down_block(32, 64),            # -> 64x64
            down_block(64, 128),           # -> 32x32
            down_block(128, 256),          # -> 16x16
            down_block(256, 512),          # -> 8x8
        )

        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.net(x)
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

        # Helper: Upsample (Transpose stride 2) -> BN -> ReLU -> ResidualBlock
        def up_block(in_c, out_c, final_layer=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            ]
            if not final_layer:
                layers.extend([
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    ResidualBlock(out_c)
                ])
            return nn.Sequential(*layers)

        # Reconstructs 256x256 from 8x8
        self.up1 = up_block(512, 256) # -> 16x16
        self.up2 = up_block(256, 128) # -> 32x32
        self.up3 = up_block(128, 64)  # -> 64x64
        self.up4 = up_block(64, 32)   # -> 128x128
        
        # Final layer to 256x256
        self.final_conv = nn.ConvTranspose2d(32, img_channels, 4, 2, 1)
        self.final_act = nn.Tanh() 

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 8, 8)
        
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        
        h = self.final_conv(h)
        return self.final_act(h)


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


def loss_function(x_hat, x, mu, logvar):
    # Reconstruction Loss (MSE)
    # Ensure reduction="sum" so it balances well with KLD sum
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld


def train_vae(vae_model, X_train, X_val, epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-5, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae_model.to(device)
    
    # Data Preparation
    # Ensure inputs are (N, 1, 256, 256)
    if X_train.ndim == 3:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer: ADAMW
    optimizer = optim.AdamW(vae_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None

    print(f"Starting VAE training on {device} for 256x256 inputs...")
    
    for epoch in range(epochs):
        vae_model.train()
        train_loss = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            x_hat, mu, logvar = vae_model(data)
            loss = loss_function(x_hat, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # Validation
        vae_model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, in val_loader:
                data = data.to(device)
                x_hat, mu, logvar = vae_model(data)
                loss = loss_function(x_hat, data, mu, logvar)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(vae_model.state_dict())

    print("Training complete.")
    
    if best_model_state is not None:
        print(f"Restoring best model (Val Loss: {best_val_loss:.4f})")
        vae_model.load_state_dict(best_model_state)
        
    return history


