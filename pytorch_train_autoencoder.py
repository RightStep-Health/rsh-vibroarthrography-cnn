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
