import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict
import torch

class GroupedIMFDataset(Dataset):
    def __init__(self, root_dir):
        self.groups = defaultdict(list)
        self.labels = {}

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith('.png'):
                    parts = fname.split('_')  # e.g. healthy_0_imf_1.png
                    label_str = parts[0]
                    signal_id = f"{label_str}_{parts[1]}"

                    full_path = os.path.join(dirpath, fname)
                    self.groups[signal_id].append(full_path)
                    self.labels[signal_id] = 0 if label_str == "healthy" else 1

        self.signal_ids = list(self.groups.keys())

        for signal_id in self.signal_ids:
            self.groups[signal_id] = sorted(
                self.groups[signal_id],
                key=lambda x: ("recon" not in x, x)
            )

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.signal_ids)

    def __getitem__(self, idx):
        signal_id = self.signal_ids[idx]
        image_paths = self.groups[signal_id]
        label = self.labels[signal_id]

        images = [self.transform(Image.open(p)) for p in image_paths]
        images = torch.stack(images, dim=0)  # shape: [4, 1, H, W]
        return images, label

import os
from pathlib import Path

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])  # optional
        ])

        root_path = Path(root_dir)
        for subdir in root_path.iterdir():
            if subdir.is_dir():
                for img_path in subdir.glob("*.png"):
                    fname = img_path.name  # e.g. "0_healthy.png"
                    parts = fname.split('_')
                    if len(parts) < 2:
                        continue  # skip unexpected filename

                    label_str = parts[1].split('.')[0].lower()  # 'healthy' or 'pathology'
                    label = 0 if label_str == 'healthy' else 1

                    self.image_paths.append(str(img_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        #print(f"Loaded image size: {image.size}")  # PIL Image size (width, height)
        image = self.transform(image)
        #print(f"Tensor shape after transform: {image.shape}")  # [C, H, W]
        label = self.labels[idx]
        return image, label
