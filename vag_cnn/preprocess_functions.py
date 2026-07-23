### utiltiy functions to pre process VAG signals
import numpy as np
from scipy.signal import resample

def double_cascaded_moving_average(signal, window_size=5):
    # First pass
    smooth1 = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    # Second pass
    smooth2 = np.convolve(smooth1, np.ones(window_size)/window_size, mode='same')
    return smooth2

# normalises between -1 and 1
def normalize_signal(signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val

def resample_to_fixed_length(signal, target_length=8192):
    return resample(signal, target_length)

def pad_to_target(signal, target_length=8192):
    signal = np.asarray(signal)
    current_length = len(signal)

    if current_length >= target_length:
        return signal[:target_length]  # trim if too long
    else:
        pad_width = target_length - current_length
        return np.pad(signal, (0, pad_width), mode='constant')

# pre processing steps
def preprocess_vag_signal(signal, window):
    n_signal = normalize_signal(signal)
    f_signal = double_cascaded_moving_average(n_signal, window_size=window)
    
    return f_signal