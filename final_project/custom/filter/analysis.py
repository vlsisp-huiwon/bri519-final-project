import numpy as np
from ..fft.fft import *

# ============================== rev(Here) ==============================
# compute_magnitude is revised custom code
def compute_magnitude(data, fs):
    N = len(data)
    freq = custom_fftfreq(N, 1/fs) 
    fft_values = huiwon_fft(data)
    return freq[:N//2], np.abs(fft_values)[:N//2]  # Return only positive vaule
# =======================================================================


def separate_tone_data(data, custom_filtered, num_sessions):
    tone_indices = []  # high- and low-tone index
    filtered_low_tones = []
    filtered_high_tones = []

    for session in range(num_sessions):
        low_tone_idx = np.where(data[session, 4] == np.min(np.unique(data[session, 4])))[0]
        high_tone_idx = np.where(data[session, 4] == np.max(np.unique(data[session, 4])))[0]
        
        tone_indices.append((low_tone_idx, high_tone_idx))
        
        # Extract high- and low-tone data from filtered data
        # ============================== rev(Here) ==============================
        # It modify custom filter
        filtered_low_tone = custom_filtered[session, low_tone_idx, :].squeeze()
        filtered_high_tone = custom_filtered[session, high_tone_idx, :].squeeze()
        # =======================================================================

        filtered_low_tones.append(filtered_low_tone)
        filtered_high_tones.append(filtered_high_tone)
    
    return filtered_low_tones, filtered_high_tones


def mean_with_std_error(data):

    # data is filtered data (trial x time point)
    mean_data = np.mean(data, axis=0)   # mean at each time point
    std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])   # standard error

    return mean_data, std_error

