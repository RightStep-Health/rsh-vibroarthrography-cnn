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

import torch

class CNNAutoencoderSpectrograms(nn.Module):
    def __init__(self, latent_dim=128, in_channels=1, image_size=(129, 125)):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # No downsampling, output: (16, 129, 125)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # Downsample by 2, output: (32, 65, 63)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # Downsample by 2, output: (64, 33, 32)
            nn.ReLU(),
        )

        # Dynamically compute flattened size after conv layers:
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *image_size)
            dummy_output = self.encoder_conv(dummy_input)
            self.flattened_size = dummy_output.numel()

        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        self.tanh = nn.Tanh()

        self.fc_dec = nn.Linear(latent_dim, self.flattened_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to (32, 65, 63)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample to (16, 129, 125)
            nn.ReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),  # Keep size
            nn.Sigmoid()  # output pixels normalized [0,1]
        )

    def forward(self, x):
        z = self.encoder_conv(x)
        x_hat = self.decoder_conv(z)
        # Crop output to input spatial size
        x_hat = x_hat[:, :, :x.size(2), :x.size(3)]
        return x_hat, z
