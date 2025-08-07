import matplotlib.pyplot as plt
import numpy as np
import preprocess_functions as preF

def plot_raw_vs_filtered_single(raw_signal, filtered_signal, label="Signal", fs=2000):


    t = np.arange(len(raw_signal)) / fs  # Time axis in seconds

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot raw
    axes[0].plot(t, raw_signal, color='gray')
    axes[0].set_title(f"{label} - Raw")
    axes[0].set_ylabel("Amplitude")

    # Plot filtered
    axes[1].plot(t, filtered_signal, color='blue')
    axes[1].set_title(f"{label} - Filtered")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_ylim(-1, 1)  # Fix y-axis to -1 to 1
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def plot_original_vs_reconstructed(signal, reconstructed, label="Signal"):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="Filtered & Normalized", alpha=0.6)
    plt.plot(reconstructed, label="Reconstructed from IMFs 3–5", linewidth=2)
    plt.title(f"{label} – Original vs Reconstructed")
    plt.legend()
    plt.tight_layout()
    plt.show()


# assumes that pre processing must be done
def plot_raw_vs_filtered(healthy_signals, pathology_signals, num_samples=5):
    """Plot raw and filtered versions of VAG signals from healthy and pathological groups."""
    fig, axes = plt.subplots(2 * num_samples, 2, figsize=(14, num_samples * 4))

    for i in range(num_samples):
        raw_healthy = healthy_signals[i]
        p_healthy = preF.preprocess_vag_signal(raw_healthy, 5)

        raw_pathology = pathology_signals[i]
        p_pathology = preF.preprocess_vag_signal(raw_pathology, 5)
        # Plot Healthy
        axes[2*i][0].plot(raw_healthy, color='gray')
        axes[2*i][0].set_title(f"Healthy #{i+1} - Raw")

        axes[2*i + 1][0].plot(p_healthy, color='green')
        axes[2*i + 1][0].set_title(f"Healthy #{i+1} - Filtered")

        # Plot Pathology
        axes[2*i][1].plot(raw_pathology, color='gray')
        axes[2*i][1].set_title(f"Pathology #{i+1} - Raw")

        axes[2*i + 1][1].plot(p_pathology, color='red')
        axes[2*i + 1][1].set_title(f"Pathology #{i+1} - Filtered")

    plt.tight_layout()
    plt.show()


def plot_imfs(signal, label, imfs, num_imfs_to_plot=7):
    num_imfs = min(num_imfs_to_plot, imfs.shape[0])

    plt.figure(figsize=(12, 2 * (num_imfs + 1)))
    plt.subplot(num_imfs + 1, 1, 1)
    plt.plot(signal, color='black')
    plt.title(f"Original Signal ({label})")
    
    for i in range(num_imfs):
        plt.subplot(num_imfs + 1, 1, i + 2)
        plt.plot(imfs[i])
        plt.title(f"IMF {i + 1}")
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_tfd(tfd, title="TFD", cmap="viridis"):
    """
    Plots a Time-Frequency Distribution (TFD) matrix as a heatmap.

    Parameters:
    - tfd: 2D numpy array of shape (time, frequency)
    - title: plot title
    - cmap: colormap (e.g., 'viridis', 'plasma', 'inferno')
    """
    plt.figure(figsize=(8, 4))
    plt.imshow(tfd.T, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label='Intensity')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()


import librosa.display

def save_qtfd_image(qtfd_matrix, save_path, sr=2000):
    print(f"saving TDF to {save_path}")
    plt.figure(figsize=(4, 2))
    librosa.display.specshow(qtfd_matrix, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()