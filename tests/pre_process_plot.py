import sys
import os
sys.path.append(os.path.abspath('..'))  # Adds parent directory to path

from data_loader import load_vag_signals
from plotting import plot_raw_vs_filtered, plot_raw_vs_filtered_single
from preprocess_functions import preprocess_vag_signal, pad_to_target

healthy_dir = "../data/open_vag/normal/"
pathology_dir = "../data/open_vag/pathology/"
h, p = load_vag_signals(healthy_dir, pathology_dir)

window_size = 25
fs=2000

r_h = pad_to_target(h[49])
r_p = pad_to_target(p[19])

p_h = preprocess_vag_signal(r_h, window_size)
p_p = preprocess_vag_signal(r_p, window_size)
#plot_raw_vs_filtered_single(r_h, p_h, "Healthy", fs)
plot_raw_vs_filtered_single(r_p, p_p, "Pathology", fs)
#plot_raw_vs_filtered(h, p)