import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

from pytorch_autoencoder import CNNAutoencoder
from pytorch_train_autoencoder import train_autoencoder
from pytorch_data_loader import GroupedIMFDataset

def extract_latent_vectors(model, dataloader, device):
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, latents = model(images)  # forward pass, get latent vectors
            all_latents.append(latents.cpu())
            all_labels.append(labels)

    all_latents = torch.cat(all_latents, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_latents, all_labels

def visualize_latent_space(latents, labels, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedding = reducer.fit_transform(latents)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f"Latent Space Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def main():
    dataset = GroupedIMFDataset("TFDs/")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNAutoencoder(latent_dim=128).to(device)

    # Train the autoencoder
    train_autoencoder(model, dataloader, device, epochs=20, lr=1e-3)

    # Extract latent vectors and labels from the dataset
    latent_vectors, labels = extract_latent_vectors(model, dataloader, device)

    # Visualize latent space clusters
    visualize_latent_space(latent_vectors, labels, method="umap")  # or "tsne"

if __name__ == "__main__":
    main()

