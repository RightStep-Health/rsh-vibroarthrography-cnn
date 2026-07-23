"""Canonical ResNet18 training entrypoint for CWT image classification.

Methodology:
- fixed random seed for reproducibility
- stratified 5-fold cross-validation
- out-of-fold predictions aggregated across all folds
- final report based on those out-of-fold predictions
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SEED = 42
DEFAULT_DATA_ROOT = "images/EMD_CWT_244/"
DEFAULT_MODEL_OUTPUT = "models/resnet_spectrogram_classifier_cv_best_fold.pth"
DEFAULT_METRICS_OUTPUT = "models/resnet_spectrogram_classifier_cv_metrics.json"
DEFAULT_PREDICTIONS_OUTPUT = "models/resnet_spectrogram_classifier_cv_oof_predictions.csv"
DEFAULT_NUM_FOLDS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SpectrogramResNet18(nn.Module):
    def __init__(self, num_classes=2, input_mode="grayscale"):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if input_mode == "grayscale":
            pretrained_weights = self.model.conv1.weight.data
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SpectrogramImageDataset(Dataset):
    def __init__(self, root_dir, input_mode):
        self.image_paths = []
        self.labels = []
        self.input_mode = input_mode
        self.class_to_idx = {"healthy": 0, "pathology": 1}

        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".png"):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        convert_mode = "L" if self.input_mode == "grayscale" else "RGB"
        image = Image.open(self.image_paths[idx]).convert(convert_mode)
        label = self.labels[idx]
        return image, label


class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label


def build_transforms(input_mode):
    train_steps = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    eval_steps = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    if input_mode == "rgb":
        imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        train_steps.append(imagenet_normalize)
        eval_steps.append(imagenet_normalize)
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(eval_steps)
    return train_transform, eval_transform


def create_dataloader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_classifier(model, dataloader, device, epochs, lr):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
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
        accuracy = correct / total
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} accuracy={accuracy:.4f}")


def collect_predictions(model, dataloader, device):
    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Pathological"],
        yticklabels=["Healthy", "Pathological"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Out-of-Fold Confusion Matrix")
    plt.tight_layout()
    plt.show()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_metrics(metrics_output, summary):
    ensure_parent_dir(metrics_output)
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to {metrics_output}")


def save_oof_predictions(predictions_output, image_paths, oof_true, oof_pred):
    ensure_parent_dir(predictions_output)
    with open(predictions_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_path", "true_label", "predicted_label"])
        for idx, (image_path, true_label, pred_label) in enumerate(zip(image_paths, oof_true, oof_pred)):
            writer.writerow([idx, image_path, int(true_label), int(pred_label)])
    print(f"Saved out-of-fold predictions to {predictions_output}")


def resolve_output_path(output_path, input_mode):
    if input_mode == "grayscale":
        return output_path
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{input_mode}{path.suffix}"))


def run_cross_validation(
    data_root,
    batch_size,
    epochs,
    lr,
    num_folds,
    seed,
    model_output,
    metrics_output,
    predictions_output,
    input_mode,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_output = resolve_output_path(model_output, input_mode)
    metrics_output = resolve_output_path(metrics_output, input_mode)
    predictions_output = resolve_output_path(predictions_output, input_mode)

    base_dataset = SpectrogramImageDataset(data_root, input_mode=input_mode)
    labels = np.array(base_dataset.labels)
    train_transform, eval_transform = build_transforms(input_mode)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_accuracies = []
    fold_f1_scores = []
    fold_summaries = []
    oof_true = np.empty(len(base_dataset), dtype=int)
    oof_pred = np.empty(len(base_dataset), dtype=int)
    best_model_state = None
    best_fold_f1 = -1.0

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n=== Fold {fold_index}/{num_folds} ===")

        train_subset = TransformDataset(Subset(base_dataset, train_idx), train_transform)
        val_subset = TransformDataset(Subset(base_dataset, val_idx), eval_transform)

        train_loader = create_dataloader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + fold_index,
        )
        val_loader = create_dataloader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed + fold_index,
        )

        model = SpectrogramResNet18(num_classes=2, input_mode=input_mode)
        train_classifier(model, train_loader, device, epochs=epochs, lr=lr)

        fold_y_true, fold_y_pred = collect_predictions(model, val_loader, device)
        fold_accuracy = accuracy_score(fold_y_true, fold_y_pred)
        fold_f1 = f1_score(fold_y_true, fold_y_pred)

        print(f"Fold {fold_index} accuracy: {fold_accuracy:.4f}")
        print(f"Fold {fold_index} pathological F1: {fold_f1:.4f}")

        fold_accuracies.append(fold_accuracy)
        fold_f1_scores.append(fold_f1)
        fold_summaries.append({
            "fold": fold_index,
            "train_size": len(train_idx),
            "validation_size": len(val_idx),
            "accuracy": float(fold_accuracy),
            "pathological_f1": float(fold_f1),
        })
        oof_true[val_idx] = fold_y_true
        oof_pred[val_idx] = fold_y_pred

        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_model_state = copy.deepcopy(model.state_dict())

    print("\n=== Cross-Validation Summary ===")
    print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    print(f"Mean pathological F1: {np.mean(fold_f1_scores):.4f} +/- {np.std(fold_f1_scores):.4f}")

    print("\n=== Out-of-Fold Classification Report ===")
    report_text = classification_report(oof_true, oof_pred, target_names=["Healthy", "Pathological"])
    print(report_text)
    plot_confusion_matrix(oof_true, oof_pred)

    summary = {
        "methodology": {
            "input_mode": input_mode,
            "seed": seed,
            "folds": num_folds,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "data_root": data_root,
        },
        "fold_metrics": fold_summaries,
        "aggregate_metrics": {
            "mean_accuracy": float(np.mean(fold_accuracies)),
            "std_accuracy": float(np.std(fold_accuracies)),
            "mean_pathological_f1": float(np.mean(fold_f1_scores)),
            "std_pathological_f1": float(np.std(fold_f1_scores)),
        },
        "out_of_fold_classification_report": report_text,
    }

    if best_model_state is not None:
        ensure_parent_dir(model_output)
        torch.save(best_model_state, model_output)
        print(f"Saved best-fold model to {model_output}")

    save_metrics(metrics_output, summary)
    save_oof_predictions(predictions_output, base_dataset.image_paths, oof_true, oof_pred)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a grayscale ResNet18 on CWT images with stratified cross-validation."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Directory containing healthy/ and pathology/ image folders.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs per fold.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for Adam.")
    parser.add_argument("--folds", type=int, default=DEFAULT_NUM_FOLDS, help="Number of stratified cross-validation folds.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for Python, NumPy, and PyTorch.")
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_OUTPUT,
        help="Path to save the best-fold model weights.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["grayscale", "rgb"],
        default="grayscale",
        help="Whether to train on grayscale or RGB-decoded images.",
    )
    parser.add_argument(
        "--metrics-output",
        default=DEFAULT_METRICS_OUTPUT,
        help="Path to save run-level metrics as JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        default=DEFAULT_PREDICTIONS_OUTPUT,
        help="Path to save out-of-fold predictions as CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_cross_validation(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_folds=args.folds,
        seed=args.seed,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        input_mode=args.input_mode,
    )


if __name__ == "__main__":
    main()
