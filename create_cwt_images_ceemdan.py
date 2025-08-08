import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files only
import matplotlib.pyplot as plt
import pywt  # For Continuous Wavelet Transform
import os

from data_loader import load_vag_signals
from preprocess_functions import preprocess_vag_signal, pad_to_target
from ceemdan import ceemdan_reconstruct_midband


def process_signal(signal, window_size):
    r_s = pad_to_target(signal)
    s_n = preprocess_vag_signal(r_s, window_size)
    return s_n


def compute_cwt_image(signal, fs=2000, wavelet='morl', num_scales=128):
    """
    Compute a CWT scalogram for the given signal.

    Parameters:
        signal (array): Input 1D signal
        fs (int): Sampling frequency
        wavelet (str): Mother wavelet (default 'morl')
        num_scales (int): Number of scales to compute

    Returns:
        scalogram (2D array): Magnitude of the CWT coefficients
        frequencies (array): Corresponding pseudo-frequencies in Hz
        time (array): Time axis in seconds
    """
    # Choose scales so higher index corresponds to lower frequency
    scales = np.arange(1, num_scales + 1)
    coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    scalogram = np.abs(coef)  # magnitude
    time = np.arange(len(signal)) / fs

    return scalogram, freqs, time


from PIL import Image
import matplotlib.cm as cm

def save_cwt_multichannel(scalogram, save_path):
    # Normalize scalogram to 0–1
    scalogram_norm = (scalogram - np.min(scalogram)) / (np.max(scalogram) - np.min(scalogram) + 1e-12)

    # Channel 1: Magma colormap
    magma_rgb = cm.get_cmap('magma')(scalogram_norm)[..., :3]  # Drop alpha
    # change to this next time magma_rgb = cm.colormaps['magma'](scalogram_norm)[..., :3]  # Drop alpha

    # Channel 2: Grayscale magnitude
    gray = np.expand_dims(scalogram_norm, axis=-1).repeat(3, axis=-1)

    # Channel 3: Log-scaled magnitude
    scalogram_log = np.log1p(scalogram)  # log(1+x)
    scalogram_log_norm = (scalogram_log - np.min(scalogram_log)) / (np.max(scalogram_log) - np.min(scalogram_log) + 1e-12)
    log_gray = np.expand_dims(scalogram_log_norm, axis=-1).repeat(3, axis=-1)

    # Stack as separate channels in a single image (H, W, 3)
    combined = np.stack([
        magma_rgb[..., 0],  # R from magma
        gray[..., 0],       # grayscale
        log_gray[..., 0]    # log-scaled grayscale
    ], axis=-1)

    # Convert to uint8 and resize
    img = (combined * 255).astype(np.uint8)
    img = Image.fromarray(img).resize((244, 244), Image.BICUBIC)

    img.save(save_path)
    print(f"Saved multichannel CWT to {save_path}")


def save_cwt_image(scalogram, freqs, time, save_path):
    print(f"saving CWT scalogram to {save_path}")
    dpi = 100
    width_px, height_px = 244, 244 # standard size for resnet #125, 129
    plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    # Convert to dB for better visual contrast
    scalogram_db = 10 * np.log10(scalogram + 1e-10)

    plt.imshow(
        scalogram_db,
        extent=[time.min(), time.max(), freqs.min(), freqs.max()],
        cmap= 'magma', #'viridis',
        aspect='auto',
        origin='lower'
    )
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    window_size = 25
    fs = 2000
    healthy_dir = "data/open_vag/normal/"
    pathology_dir = "data/open_vag/pathology/"

    print("loading vag signals")
    h_signals, p_signals = load_vag_signals(healthy_dir, pathology_dir)

    base_save_dir = "EMD_CWT_244/"

    save_dir = os.path.join(base_save_dir, "healthy")
    os.makedirs(save_dir, exist_ok=True)

    for idx, s in enumerate(h_signals):
        file_name = f"{idx}_healthy.png"
        fpath = os.path.join(save_dir, file_name)
        p_s = process_signal(s, window_size)
        _, recon = ceemdan_reconstruct_midband(p_s, imf_range=(2, 5))
        scalogram, freqs, time = compute_cwt_image(recon, fs=fs)
        print("Scalogram shape:", scalogram.shape)
        #save_cwt_image(scalogram, freqs, time, fpath)
        save_cwt_multichannel(scalogram, fpath)
    save_dir = os.path.join(base_save_dir, "pathology")
    os.makedirs(save_dir, exist_ok=True)

    for idx, s in enumerate(p_signals):
        file_name = f"{idx}_pathology.png"
        fpath = os.path.join(save_dir, file_name)
        p_s = process_signal(s, window_size)
        _, recon = ceemdan_reconstruct_midband(p_s, imf_range=(2, 5))
        scalogram, freqs, time = compute_cwt_image(recon, fs=fs)
        print("Scalogram shape:", scalogram.shape)
        
        #save_cwt_image(scalogram, freqs, time, fpath) single channel color png
        save_cwt_multichannel(scalogram, fpath)



if __name__ == "__main__":
    main()
