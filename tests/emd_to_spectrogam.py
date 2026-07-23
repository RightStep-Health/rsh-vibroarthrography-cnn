
# test to go through the following pipeline

# raw vag -> EMD (CEEDMAN) -> reconstructed signal -> spectrogram


# Add parent directory to path
import sys
import os
import numpy as np
from scipy.signal import stft, get_window
sys.path.append(os.path.abspath('..'))

from vag_cnn.data_loader import load_vag_signals
from vag_cnn.plotting import plot_original_vs_reconstructed, plot_spectrogram
from vag_cnn.preprocess_functions import preprocess_vag_signal, pad_to_target
from vag_cnn.ceemdan import ceemdan_reconstruct_midband

def process_signals(signals, window_size):
    processed = []
    for s in signals:
        r_s = pad_to_target(s)
        s_n = preprocess_vag_signal(r_s, window_size)
        processed.append(s_n)
    return processed

def main():
    healthy_dir = "../data/open_vag/normal/"
    pathology_dir = "../data/open_vag/pathology/"
    
    h, p = load_vag_signals(healthy_dir, pathology_dir)

    window_size = 25  # double-cascaded moving average filter window
    p_h = process_signals(h, window_size)
    p_p = process_signals(p, window_size)

    # Use subject 49 (healthy) and 19 (pathology) as per paper
    h_test = p_h[49]
    p_test = p_p[19]

    # use epsilon value of 0.1 to kind of match paper
    h_imfs, h_recon = ceemdan_reconstruct_midband(h_test, imf_range=(2, 5))
    p_imfs, p_recon = ceemdan_reconstruct_midband(p_test, imf_range=(2, 5))

    plot_original_vs_reconstructed(h_test, h_recon, label="Healthy")
    plot_original_vs_reconstructed(p_test, p_recon, label="Pathology")

    def compute_stft_image(signal, fs=2000, window_size=256, hop_size=64):
        window = get_window('hann', window_size)
        f, t, Zxx = stft(
            signal,
            fs=fs,
            window=window,
            nperseg=window_size,
            noverlap=window_size //2,
            nfft=256,
            return_onesided=True,
            padded=False,
            boundary=None
        )

        spectrogram = np.abs(Zxx)  # magnitude of complex STFT
        return spectrogram, f, t

    # compute a spectrogram from the reconstrucuted signal
    spec, f, t = compute_stft_image(h_recon)
    plot_spectrogram(spec,f,t, title="STFT Reconstructed Healthy Spectrogram")
    spec, f, t = compute_stft_image(p_recon)
    plot_spectrogram(spec,f,t, title="STFT Reconstructed Pathology Spectrogram")

if __name__ == "__main__":
    main()
