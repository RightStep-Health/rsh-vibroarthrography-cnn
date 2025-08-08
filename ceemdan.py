from PyEMD import CEEMDAN
import numpy as np


def apply_ceemdan(signal, noise_seed=42):
    ceemdan = CEEMDAN()
    #ceemdan.noise_seed(noise_seed)
    IMFs = ceemdan(signal)
    return IMFs


def compute_ceemdan_imfs(x, trials=50, epsilon=0.1, seed=None):
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=True)
    if seed is not None:
        ceemdan.noise_seed(seed)

    imfs = ceemdan.ceemdan(x)
    residue = x - np.sum(imfs, axis=0)
    return imfs, residue

def ceemdan_reconstruct_midband(signal, imf_range=(2, 5)):
    print("computing IMFs")
    imfs, _ = compute_ceemdan_imfs(signal, seed=42)

    start, end = imf_range
    selected_imfs = imfs[start:end]  # IMFs in the desired band

    print("reconstructing signal..")
    reconstructed = np.sum(selected_imfs, axis=0)

    return list(selected_imfs), reconstructed  # Return as list for consistency

