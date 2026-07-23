# test the group data loader that reads for images per signal

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath('.'))

from vag_cnn.pytorch_data_loader import GroupedIMFDataset
from torch.utils.data import DataLoader

dataset = GroupedIMFDataset("images/TFDs/")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for signal_images, signal_labels in dataloader:
    print(signal_images.shape)  # [16, 4, 1, H, W]
    print(signal_labels.shape)  # [16]
    break
