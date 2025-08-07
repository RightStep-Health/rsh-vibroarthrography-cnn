import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
from tqdm import tqdm

from pytorch_autoencoder import CNNAutoencoderSpectrograms
from pytorch_data_loader import SpectrogramDataset


# --------------------------
# TRAINING FUNCTION
# --------------------------
def train_autoencoder_spectrograms(model, dataloader, device, epochs=20, lr=1e-3):
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Reshape: [B, 4, 1, H, W] → [B, 4, H, W]
            images = images.squeeze(2).to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

    print("Training complete!")


# --------------------------
# EXTRACT LATENT VECTORS
# --------------------------
def extract_latent_vectors(model, dataloader, device):
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.squeeze(2).to(device)
            _, latents = model(images)

            # Flatten latents if needed (e.g., from [B, C, H, W] to [B, C*H*W])
            if latents.dim() > 2:
                latents = latents.view(latents.size(0), -1)

            all_latents.append(latents.cpu())
            all_labels.append(labels)

    all_latents = torch.cat(all_latents, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_latents, all_labels


# --------------------------
# VISUALIZE LATENT SPACE
# --------------------------
def visualize_latent_space(latents, labels, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init='random', perplexity=30, max_iter=1000)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedding = reducer.fit_transform(latents)

    sil_score = silhouette_score(latents, labels)
    print(f"Silhouette Score (original latent space): {sil_score:.4f}")
    sil_score_2d = silhouette_score(embedding, labels)
    print(f"Silhouette Score (2D {method.upper()} embedding): {sil_score_2d:.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f"Latent Space Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


# --------------------------
# MAIN
# --------------------------
def main():
    dataset = SpectrogramDataset("SPECS/")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace with actual spectrogram dimensions if needed
    H, W = 129, 125
    model = CNNAutoencoderSpectrograms(in_channels=1, image_size=(H, W)).to(device)

    # Train the autoencoder
    train_autoencoder_spectrograms(model, dataloader, device, epochs=20, lr=1e-3)

    # Extract latent representations
    latent_vectors, labels = extract_latent_vectors(model, dataloader, device)

    # Visualize latent space
    visualize_latent_space(latent_vectors, labels, method="umap")  # or "tsne"


if __name__ == "__main__":
    main()
