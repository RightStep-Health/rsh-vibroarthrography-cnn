import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict
import torch

class GroupedIMFDataset(Dataset):
    def __init__(self, root_dir):
        self.groups = defaultdict(list)  # key: signal ID, value: list of image paths
        self.labels = {}  # key: signal ID, value: label (0=healthy, 1=pathology)

        for fname in os.listdir(root_dir):
            if fname.endswith('.png'):
                parts = fname.split('_')  # e.g. healthy_0_imf_1.png → ['healthy', '0', 'imf', '1.png']
                label_str = parts[0]
                signal_id = f"{label_str}_{parts[1]}"  # e.g. healthy_0

                full_path = os.path.join(root_dir, fname)
                self.groups[signal_id].append(full_path)
                self.labels[signal_id] = 0 if label_str == "healthy" else 1

        self.signal_ids = list(self.groups.keys())

        # Sort images within each signal group
        for signal_id in self.signal_ids:
            self.groups[signal_id] = sorted(
                self.groups[signal_id],
                key=lambda x: ("recon" not in x, x)  # ensures recon comes last
            )

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
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
