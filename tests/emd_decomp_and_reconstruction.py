import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath('..'))

from data_loader import load_vag_signals
from plotting import plot_original_vs_reconstructed
from preprocess_functions import preprocess_vag_signal, pad_to_target
from ceemdan import ceemdan_reconstruct_midband

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

if __name__ == "__main__":
    main()
