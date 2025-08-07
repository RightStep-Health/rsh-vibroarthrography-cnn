import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (B,16,H/2,W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (B,32,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (B,64,H/8,W/8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (H//8) * (W//8), latent_dim),
            nn.Tanh()  # map latent to [-1,1]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * (H//8) * (W//8)),
            nn.Unflatten(1, (64, H//8, W//8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # output normalized [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# Replace H and W with your image height and width from dataset
H, W = 128, 256  # or swap depending on your saved shape

