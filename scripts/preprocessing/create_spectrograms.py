import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vag_cnn.data_loader import load_vag_signals
from vag_cnn.preprocess_functions import preprocess_vag_signal, pad_to_target

def process_signal(signal, window_size):
    r_s = pad_to_target(signal)
    s_n = preprocess_vag_signal(r_s, window_size)
    return s_n

def compute_stft_image(signal, fs=2000, window_size=256, hop_size=64):
    window = get_window('hann', window_size)
    f, t, Zxx = stft(
        signal,
        fs=fs,
        window=window,
        nperseg=window_size,
        noverlap=window_size - hop_size,
        nfft=256,
        return_onesided=True,
        padded=False,
        boundary=None
    )

    spectrogram = np.abs(Zxx)  # magnitude of complex STFT
    return spectrogram

def save_spectrogram_image(spec, save_path):
    print(f"saving TDF to {save_path}")
    dpi = 100  # dots per inch
    width_px, height_px = 125, 129  # width, height in pixels
    plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    plt.imshow(10 * np.log10(spec + 1e-10), aspect='auto', origin='lower', cmap='viridis')
    #plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

window_size = 10
fs=2000
healthy_dir = "data/open_vag/normal/"
pathology_dir = "data/open_vag/pathology/"

print("loading vag signals")
h_signals, p_signals = load_vag_signals(healthy_dir, pathology_dir)

base_save_dir = "images/SPECS"

save_dir = os.path.join(base_save_dir, "healthy")
os.makedirs(save_dir, exist_ok=True)
for idx, s in enumerate(h_signals):
    file_name = f"{idx}_healthy.png"
    fpath = os.path.join(save_dir, file_name)
    p_s = process_signal(s, window_size)
    spec = compute_stft_image(p_s, fs=fs)
    print("Spectrogram shape:", spec.shape)  # Expect (129, 126)
    save_spectrogram_image(spec, fpath)

save_dir = os.path.join(base_save_dir, "pathology")
os.makedirs(save_dir, exist_ok=True)
for idx, s in enumerate(p_signals):
    file_name = f"{idx}_pathology.png"
    fpath = os.path.join(save_dir, file_name)
    p_s = process_signal(s, window_size)
    spec = compute_stft_image(p_s, fs=fs)
    print("Spectrogram shape:", spec.shape)  # Expect (129, 126)
    save_spectrogram_image(spec, fpath)

