import torch
import torch.nn as nn
import torch.nn.functional as F

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
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld

'''
vae = VAE(latent_dim=64, img_channels=1).cuda()
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(80):
    for x in dataloader:
        x = x.cuda()
        x_hat, mu, logvar = vae(x)

        loss = loss_function(x_hat, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.2f}")
'''