import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vag_cnn.data_loader import load_vag_signals
from vag_cnn.plotting import save_qtfd_image
from vag_cnn.preprocess_functions import preprocess_vag_signal, pad_to_target
from vag_cnn.ceemdan import ceemdan_reconstruct_midband
from vag_cnn.tfds import compute_qtfd

def process_signal(signal, window_size):
    r_s = pad_to_target(signal)
    s_n = preprocess_vag_signal(r_s, window_size)
    return s_n

def process_group(signals, label, window_size, base_save_dir):
    print(f"processing {label} signals")
    save_dir = os.path.join(base_save_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    for idx, s in enumerate(signals):
        print(f"Processing {label} signal {idx}")
        p_s = process_signal(s, window_size)

        print("computing ceemdan and reconstrucing signal")
        imfs, recon = ceemdan_reconstruct_midband(p_s, imf_range=(2, 5))
        imfs.append(recon)  # Add the reconstructed signal as the final item

        print("computing TFD")
        for i, imf in enumerate(imfs):
            qtfd_image = compute_qtfd(imf, fs=2000)

            if i < len(imfs) - 1:
                fname = f"{label}_{idx}_imf_{i}.png"
            else:
                fname = f"{label}_{idx}_recon.png"

            fpath = os.path.join(save_dir, fname)
            save_qtfd_image(qtfd_image, fpath, sr=2000)

def main():
    base_save_dir = "images/TFDs"
    os.makedirs(base_save_dir, exist_ok=True)

    window_size = 25
    healthy_dir = "data/open_vag/normal/"
    pathology_dir = "data/open_vag/pathology/"

    print("loading vag signals")
    h_signals, p_signals = load_vag_signals(healthy_dir, pathology_dir)

    process_group(h_signals, label="healthy", window_size=window_size, base_save_dir=base_save_dir)
    print("✅ Healthy TFDs saved")

    process_group(p_signals, label="pathology", window_size=window_size, base_save_dir=base_save_dir)
    print("✅ Pathology TFDs saved")

if __name__ == "__main__":
    main()
