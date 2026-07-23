"""Legacy cross-validation ResNet18 experiment."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --------------------------
# Model: ResNet18 for Spectrograms with pretrained weights adapted to 1-channel
# --------------------------
class SpectrogramResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(SpectrogramResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        pretrained_weights = self.model.conv1.weight.data
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------
# Training Loop
# --------------------------
def train_classifier(model, dataloader, device, epochs=20, lr=1e-5):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")


# --------------------------
# Evaluation (return metrics)
# --------------------------
def evaluate_model(model, dataloader, device, plot_cm=True):
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

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Healthy", "Pathological"])
    
    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Healthy", "Pathological"], yticklabels=["Healthy", "Pathological"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return acc, report


# --------------------------
# Main with Cross Validation
# --------------------------
def main_cv():
    data_root = "images/EMD_CWT_244/"
    batch_size = 16
    epochs = 30
    lr = 1e-4
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalize can be added if you want, test with/without
    ])

    full_dataset = SpectrogramImageDataset(data_root, transform=transform)
    labels = np.array(full_dataset.labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n=== Fold {fold + 1} ===")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = SpectrogramResNet18(num_classes=num_classes)
        model.to(device)

        train_classifier(model, train_loader, device, epochs=epochs, lr=lr)

        acc, report = evaluate_model(model, val_loader, device)
        print(report)

        fold_accuracies.append(acc)
        fold_reports.append(report)

    print(f"\nAverage accuracy across folds: {np.mean(fold_accuracies):.4f}")


if __name__ == "__main__":
    main_cv()
