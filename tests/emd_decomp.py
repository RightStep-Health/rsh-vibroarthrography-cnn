import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath('..'))

from data_loader import load_vag_signals
from plotting import plot_imfs
from preprocess_functions import preprocess_vag_signal, pad_to_target
from ceemdan import compute_ceemdan_imfs

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
    imfs_h, _ = compute_ceemdan_imfs(h_test, seed=42)
    imfs_p, _ = compute_ceemdan_imfs(p_test, seed=42)

    plot_imfs(h_test, "Healthy", imfs_h, num_imfs_to_plot=7)
    plot_imfs(p_test, "Pathology", imfs_p, num_imfs_to_plot=7)

if __name__ == "__main__":
    main()
