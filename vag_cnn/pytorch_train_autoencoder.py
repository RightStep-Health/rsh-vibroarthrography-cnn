import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # nice progress bar

def train_autoencoder(model, dataloader, device, epochs=10, lr=1e-3):
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(images)       # Forward pass
            loss = criterion(outputs, images) # Reconstruction loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.6f}")

    print("Training complete!")



def train_autoencoder_spectrogams(model, dataloader, device, epochs=10, lr=1e-3):
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # If images are batched as [B, 1, H, W], no changes needed
            # If images are stacked across IMF dimension as [B, 4, 1, H, W], adjust:
            if images.dim() == 5:
                # Collapse to [B, C, H, W] where C = 4 (IMFs)
                images = images.squeeze(2)  # from [B, 4, 1, H, W] -> [B, 4, H, W]

            images = images.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.6f}")

    print("Training complete!")
