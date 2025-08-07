import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, in_channels=4, image_size=(200, 400)):
        super().__init__()
        H, W = image_size  # unpack height and width

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # (B,16,H/2,W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),           # (B,32,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # (B,64,H/8,W/8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (H // 8) * (W // 8), latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * (H // 8) * (W // 8)),
            nn.Unflatten(1, (64, H // 8, W // 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
