# test the group data loader that reads for images per signal

from pytorch_data_loader import GroupedIMFDataset
from torch.utils.data import DataLoader

dataset = GroupedIMFDataset("TFDs/")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for signal_images, signal_labels in dataloader:
    print(signal_images.shape)  # [16, 4, 1, H, W]
    print(signal_labels.shape)  # [16]
    break
