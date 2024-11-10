from .fft import *

# ============================== rev-add(Here) ==============================
def custom_spectrogram(signal, fs, wind, overl, nfft=None):

    step = wind - overl
    num_wind = (len(signal) - overl) // step

    hann_wind = 0.5 - 0.5 * np.cos((2*np.pi*np.arange(wind)) / (wind - 1))
   
    if nfft is None:
        freq = np.array(custom_rfftfreq(wind, 1/fs))
    else:
        freq = np.array(custom_rfftfreq(nfft, 1/fs))
    
    time = np.arange(0, num_wind*step, step) / fs

    Sxx = np.zeros((len(freq), num_wind))

    for i in range(num_wind):
        start = i * step
        segment = signal[start : start + wind]

        window_segment = segment * hann_wind

        if nfft is None:
            fft_result = huiwon_fft(window_segment)
            spectrum = fft_result[:len(window_segment) // 2 + 1]
            Sxx[:, i] = (np.abs(spectrum)**2)

        else:
            spectrum = custom_rfft(window_segment, nfft)[:len(freq)]
            Sxx[:, i] = (np.abs(spectrum)**2) / (fs * np.sum(hann_wind**2))

    return freq, time, Sxx
# =======================================================================
