import numpy as np
from scipy.signal import hilbert
from tftb.processing import WignerVilleDistribution

def separable_kernel_smooth(W, doppler_kernel, lag_kernel):
    # Smooth over time (axis=1) with Doppler kernel
    smooth_time = np.apply_along_axis(
        lambda m: np.convolve(m, doppler_kernel, mode='same'),
        axis=1, arr=W
    )
    # Smooth over frequency (axis=0) with Lag kernel
    smooth_freq_time = np.apply_along_axis(
        lambda m: np.convolve(m, lag_kernel, mode='same'),
        axis=0, arr=smooth_time
    )
    return smooth_freq_time


def block_average(mat, block_size_freq, block_size_time):
    new_shape = (mat.shape[0]//block_size_freq, block_size_freq,
                 mat.shape[1]//block_size_time, block_size_time)
    return mat.reshape(new_shape).mean(axis=(1,3))

def compute_qtfd(signal, fs=2000, out_shape=(256, 128),
                 doppler_len=256, lag_len=128):
    """
    Computes QTFD using separable kernel smoothed WVD, then downsamples it.

    Parameters:
        signal (np.ndarray): 1D input signal.
        fs (int): Sampling rate.
        out_shape (tuple): Final shape (time, freq).
        doppler_len (int): Doppler kernel length (time smoothing).
        lag_len (int): Lag kernel length (frequency smoothing).

    Returns:
        np.ndarray: QTFD image of shape out_shape (default: 256×128).
    """
    # Step 1: Analytic signal
    analytic = hilbert(signal)

    # Step 2: WVD computation
    wvd = WignerVilleDistribution(analytic)
    W, _, _ = wvd.run()

    # Step 3: Kernel creation
    doppler_kernel = np.hanning(doppler_len)
    lag_kernel = np.hanning(lag_len)

    # Step 4: Apply separable smoothing
    QTFD_full = separable_kernel_smooth(W, doppler_kernel, lag_kernel)

    # Step 5: Downsample to desired shape
    freq_factor = QTFD_full.shape[0] // out_shape[1] # lag_len
    time_factor = QTFD_full.shape[1] // out_shape[0] # doppler len
         
    QTFD_downsampled = block_average(QTFD_full, freq_factor, time_factor)

    # Optional: transpose to (time, freq)
    return QTFD_downsampled.T  # shape (time, freq)
