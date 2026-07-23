"""Legacy autoencoder-plus-ResNet experiment."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152, ResNet152_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Dataset class from directory structure
# --------------------------
class SpectrogramImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {"healthy": 0, "pathology": 1}

        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        #image = Image.open(img_path).convert("L")  # Grayscale

        image = Image.open(img_path).convert("RGB")  # load as 3-channel RGB

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------
# Autoencoder architecture (simple example)
# --------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            #nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> Bx16x112x112
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # change input channels to 3
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> Bx32x56x56
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),                      # -> Bx64x50x50 (adjust if needed)
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),             # -> Bx32x56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> Bx16x112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # -> Bx1x224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# --------------------------
# Combined model: AE encoder + ResNet152 classifier
# --------------------------
class AE_ResNet152_Classifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder

        weights = ResNet152_Weights.DEFAULT
        self.resnet = resnet152(weights=weights)

        # Adapter: convert AE encoder output channels to 3 and upsample to 224x224 for ResNet input
        self.input_adapt = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )

        # Replace final classifier layer for your task
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        adapted = self.input_adapt(features)
        out = self.resnet(adapted)
        return out


# --------------------------
# Train autoencoder
# --------------------------
def train_autoencoder(model, dataloader, device, epochs=20, lr=1e-3):
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in tqdm(dataloader, desc=f"AE Epoch {epoch+1}/{epochs}"):
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"AE Epoch {epoch+1}: Loss = {avg_loss:.6f}")


# --------------------------
# Train classifier
# --------------------------
def train_classifier(model, dataloader, device, criterion, optimizer, epochs=20):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Classifier Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Classifier Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")


# --------------------------
# Evaluate classifier
# --------------------------
def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Healthy", "Pathological"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Healthy", "Pathological"], yticklabels=["Healthy", "Pathological"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# --------------------------
# Main
# --------------------------
def main():
    data_root = "images/EMD_CWT/"
    batch_size = 16
    ae_epochs = 30
    cls_epochs = 30
    ae_lr = 1e-3
    cls_lr = 1e-5
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SpectrogramImageDataset(data_root, transform=transform)

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42,
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader_ae = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # --- 1) Train Autoencoder ---
    ae_model = Autoencoder()
    print("Training Autoencoder...")
    train_autoencoder(ae_model, train_loader_ae, device, epochs=ae_epochs, lr=ae_lr)

    # --- 2) Use AE encoder in classifier ---
    classifier_model = AE_ResNet152_Classifier(ae_model.encoder, num_classes=num_classes)

    # Optional: freeze AE encoder weights or leave trainable
    # for param in classifier_model.encoder.parameters():
    #     param.requires_grad = False

    # Compute class weights (optional)
    labels = dataset.labels[train_idx]
    class_sample_counts = np.bincount(labels)
    class_weights = 1. / class_sample_counts
    class_weights = class_weights / class_weights.sum() * len(class_sample_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(classifier_model.parameters(), lr=cls_lr)

    train_loader_cls = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    print("Training Classifier...")
    train_classifier(classifier_model, train_loader_cls, device, criterion, optimizer, epochs=cls_epochs)

    print("Evaluating Classifier...")
    evaluate(classifier_model, val_loader, device)

    torch.save(classifier_model.state_dict(), "ae_resnet152_classifier.pth")


if __name__ == "__main__":
    main()
