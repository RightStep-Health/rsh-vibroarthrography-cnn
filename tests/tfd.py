import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath('..'))

from vag_cnn.data_loader import load_vag_signals
from vag_cnn.plotting import plot_tfd
from vag_cnn.preprocess_functions import preprocess_vag_signal, pad_to_target
from vag_cnn.ceemdan import ceemdan_reconstruct_midband
from vag_cnn.tfds import compute_qtfd

def process_signals(signals, window_size):
    processed = []
    for s in signals:
        r_s = pad_to_target(s)
        print("resampled", len(r_s))
        s_n = preprocess_vag_signal(r_s, window_size)
        processed.append(s_n)
    return processed

def main():
    healthy_dir = "../data/open_vag/normal/"
    pathology_dir = "../data/open_vag/pathology/"
    
    h, p = load_vag_signals(healthy_dir, pathology_dir)

    # Use subject 49 (healthy) and 19 (pathology) as per paper
    h_test = h[49]
    p_test = p[19]

    window_size = 25  # double-cascaded moving average filter window
    p_h = process_signals([h_test], window_size)[0]
    p_p = process_signals([p_test], window_size)[0]


    # use epsilon value of 0.1 to kind of match paper
    print("computing CEEMDAN and reconstructing signal")
    h_imfs, h_recon = ceemdan_reconstruct_midband(p_h, imf_range=(2, 5))
    p_imfs, p_recon = ceemdan_reconstruct_midband(p_p, imf_range=(2, 5))

    # remember to compute these for IMFs also
    print("reconstructed signals length", len(h_recon), len(p_recon))
    qtfd_image = compute_qtfd(h_recon, fs=2000)
    print(qtfd_image.shape)  # Should be (256, 128)

    plot_tfd(qtfd_image)

if __name__ == "__main__":
    main()
