"""Legacy ResNet18 experiment with class weights and threshold tuning."""

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
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,  recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------

#Class weights calculated from dataset label distribution and used in CrossEntropyLoss.

#Augmentation tweaks: Add ColorJitter to your transforms.#

#Threshold tuning: Instead of fixed argmax on outputs, apply sigmoid on the "Pathological" class output probs and vary threshold (for binary classification) — printing accuracy, recall, F1 at different thresholds.

# -----------------

# --------------------------
# Model: ResNet18 for Spectrograms with pretrained weights adapted to 1-channel
# --------------------------
class SpectrogramResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(SpectrogramResNet18, self).__init__()
        # Load pretrained weights for ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Get pretrained conv1 weights (shape: [64, 3, 7, 7])
        pretrained_weights = self.model.conv1.weight.data
        
        # Replace conv1 to accept 1 input channel (grayscale)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize conv1 weights by averaging pretrained weights across RGB channels
        self.model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

        # Replace the classifier head for your number of classes
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

        self.labels = np.array(self.labels)  # For easier computation

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
def train_classifier(model, dataloader, device, class_weights,  epochs=20, lr=1e-5):
    model.to(device)
    model.train()

    # added class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")


# --------------------------
# Evaluation
# --------------------------
def evaluate_model_with_thresholds(model, dataloader, device):
    model.eval()
    model.to(device)

    y_true = []
    y_scores = []  # Store softmax probs for pathological class

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # shape [batch, 2]
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for "pathology" class

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Threshold tuning loop
    thresholds = np.arange(0.1, 0.91, 0.1)
    print("\nThreshold tuning results (Pathological class probability cutoff):")
    print(f"{'Thresh':>6} {'Accuracy':>8} {'Recall':>8} {'F1':>8}")

    best_thresh = 0.5
    best_f1 = 0
    for thresh in thresholds:
        y_pred_thresh = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred_thresh)
        recall = recall_score(y_true, y_pred_thresh)
        f1 = f1_score(y_true, y_pred_thresh)

        print(f"{thresh:6.2f} {acc:8.3f} {recall:8.3f} {f1:8.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nBest threshold by F1 score: {best_thresh:.2f} with F1 = {best_f1:.3f}")

    # Final classification report and confusion matrix at best threshold
    y_pred_final = (y_scores >= best_thresh).astype(int)

    print("\n🧪 Classification Report at best threshold:")
    print(classification_report(y_true, y_pred_final, target_names=["Healthy", "Pathological"]))

    cm = confusion_matrix(y_true, y_pred_final)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Healthy", "Pathological"], yticklabels=["Healthy", "Pathological"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (threshold={best_thresh:.2f})")
    plt.tight_layout()
    plt.show()


# --------------------------
# Main
# --------------------------
def main():
    data_root = "images/EMD_CWT/"
    batch_size = 16
    epochs = 30
    lr = 1e-5
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Augmentations including ColorJitter
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])

    full_dataset = SpectrogramImageDataset(data_root, transform=transform)

    # Compute class weights: inverse frequency
    labels = full_dataset.labels
    class_sample_counts = np.bincount(labels)
    class_weights = 1.0 / class_sample_counts
    class_weights = class_weights / class_weights.sum() * len(class_sample_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights.numpy()}")

    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = SpectrogramResNet18(num_classes=num_classes)

    train_classifier(model, train_loader, device, class_weights, epochs=epochs, lr=lr)

    evaluate_model_with_thresholds(model, val_loader, device)

    torch.save(model.state_dict(), "resnet_spectrogram_classifier_weighted.pth")


if __name__ == "__main__":
    main()
