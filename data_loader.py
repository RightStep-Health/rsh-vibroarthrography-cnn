
import os

# load signals from two specified directories
def load_vag_signals(healthy_dir, pathology_dir):
    healthy_signals = []
    pathology_signals = []

    healthy_files = [f for f in os.listdir(healthy_dir) if '.ipynb_checkpoints' not in f]
    pathology_files = [f for f in os.listdir(pathology_dir) if '.ipynb_checkpoints' not in f]

    print(f"Loaded {len(healthy_files)} healthy VAG signals")
    print(f"Loaded {len(pathology_files)} pathology VAG signals")

    max_size_files = max(len(healthy_files), len(pathology_files))

    for i in range(max_size_files):
        if i < len(healthy_files):
            file_path = os.path.join(healthy_dir, healthy_files[i])
            with open(file_path, 'r') as file:
                vags = [float(x) for x in file.read().split()]
                healthy_signals.append(vags)

        if i < len(pathology_files):
            file_path = os.path.join(pathology_dir, pathology_files[i])
            with open(file_path, 'r') as file:
                vags = [float(x) for x in file.read().split()]
                pathology_signals.append(vags)

    return healthy_signals, pathology_signals