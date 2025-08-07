import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

from pytorch_autoencoder import CNNAutoencoder
from pytorch_train_autoencoder import train_autoencoder
from pytorch_data_loader import GroupedIMFDataset


from sklearn.metrics import silhouette_score

def extract_latent_vectors(model, dataloader, device):
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            # Reshape: [B, 4, 1, H, W] → [B, 4, H, W]
            images = images.squeeze(2).to(device)

            _, latents = model(images)  # forward pass
            all_latents.append(latents.cpu())
            all_labels.append(labels)

    all_latents = torch.cat(all_latents, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_latents, all_labels


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def visualize_latent_space(latents, labels, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init='random', perplexity=30, max_iter=1000)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedding = reducer.fit_transform(latents)

    sil_score = silhouette_score(latents, labels)
    print(f"Silhouette Score: {sil_score:.4f}")
    sil_score_2d = silhouette_score(embedding, labels)
    print(f"Silhouette score on 2D embedding: {sil_score_2d:.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f"Latent Space Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def train_autoencoder(model, dataloader, device, epochs=20, lr=1e-5):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, _ in dataloader:
            # Reshape: [B, 4, 1, H, W] → [B, 4, H, W]
            images = images.squeeze(2).to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


def main():
    dataset = GroupedIMFDataset("TFDs/")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace H and W based on your actual image size
    H, W = 200, 400
    model = CNNAutoencoder(latent_dim=128, in_channels=4, image_size=(H, W)).to(device)

    # Train the autoencoder
    train_autoencoder(model, dataloader, device, epochs=20, lr= 1e-3)#1e-3)

    # Extract latent vectors and labels from the dataset
    latent_vectors, labels = extract_latent_vectors(model, dataloader, device)

    # Visualize latent space clusters
    visualize_latent_space(latent_vectors, labels, method="tsne")  # or "umap"


if __name__ == "__main__":
    main()
