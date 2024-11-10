import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.io
from scipy.signal import butter, cheby1, cheby2, filtfilt

from custom.fft.fft import *
from custom.fft.plot import *
from custom.fft.spectrogram import *
from custom.filter.filter import *
from custom.filter.plot.plot import *

plt.rcParams['figure.max_open_warning'] = 50

# Question 1
# 1. Define parameters
fs = 1000   #sampling freq.
dur = 2     # signal duration
f_start = 100   # start freq.
f_end = 200     # end freq.

# 2. Create a time base
time = np.linspace(0, dur, fs*dur)
sig = sg.chirp(time, f_start, dur/2, f_end, 'quadratic')

# 3. Implement low level fft
N = len(sig)

# 3-1. padding the signal and time
def padding_time(N):
    power = 1
    while power < N:
        power *= 2
    return power

padded_time = padding_time(N)
padded_signal = np.pad(sig, (0, padded_time - N), mode='constant')

# 3-3. Perform FFT using the custom huiwon_fft func.
fft_huiwon = huiwon_fft(padded_signal)

comparison_of_lib_fft(fs, time, sig, padded_signal, fft_huiwon)

wind = 256
overl = wind - 1

f, tt, Sxx = custom_spectrogram(sig, fs, wind, overl)

# 9. Plot the spectrogram
plt.figure(figsize=(12, 2))
plt.pcolormesh(tt, f, 10 * np.log10(Sxx), shading='gouraud')
plt.colorbar(label='Intensity [dB]')
plt.title('Spectrogram of Chirp Signal')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.ylim(0, fs / 2)  # Limit y-axis to half the sampling frequency
plt.savefig('./result/question 1/Spectrogram of Chirp Signal.png')
plt.close() 

# Question 2
cutoff_freq = 1000  # low pass cutoff freq.
bin_width = 5       # width of freq. bins
max_freq = 200      # max. freq. to plot
num_sessions = 4    # number of sessions in the dataset
num_trials = 200    # number of trials in the dataset
fs = 10000          # sampling frequency

# Load the dataset
mat_data = scipy.io.loadmat('./mouseLFP.mat')
data = mat_data['DATA']

data_samples = data[0, 0].shape[1]

# Apply low-pass filter across sessions and trials
custom_filtered = filtfilter(data, num_sessions, num_trials, data_samples, cutoff_freq, fs)

# The result of specific session and trial 
session_idx = 0
trial_idx = 0

original_data = data[session_idx, 0][trial_idx]
filtered_data = custom_filtered[session_idx, trial_idx]

# Visualize the filtering result
plot_time_and_frequency_response(original_data, filtered_data, fs, title=f"Session {session_idx + 1} Trial {trial_idx + 1}", save_path=f'./result/question 2/Custom Filter_session_{session_idx+1}_trial_{trial_idx+1}_frequency_response.png')
# <Mean of all trials>
plot_time_and_frequency_response_mean(data, custom_filtered, fs, num_sessions, num_trials, data_samples, label="Custom Filter")

# Define constants
nyquist_freq = fs / 2

# Butterworth, Chebyshev Type I and Type II filters coefficients
butter_b, butter_a = butter(5, cutoff_freq / nyquist_freq, btype='low')
cheby1_b, cheby1_a = cheby1(5, 0.2, cutoff_freq / nyquist_freq, btype='low')
cheby2_b, cheby2_a = cheby2(5, 0.2, cutoff_freq / nyquist_freq, btype='low')

butter_filtered = np.zeros((num_sessions, num_trials, data_samples)) 
cheby1_filtered = np.zeros((num_sessions, num_trials, data_samples))
cheby2_filtered = np.zeros((num_sessions, num_trials, data_samples))

# Apply filter across sessions and trials
for session in range(num_sessions):
    for trial in range(num_trials):
        butter_filtered[session, trial, :] = filtfilt(butter_b, butter_a, data[session, 0][trial])
        cheby1_filtered[session, trial, :] = filtfilt(cheby1_b, cheby1_a, data[session, 0][trial])
        cheby2_filtered[session, trial, :] = filtfilt(cheby2_b, cheby2_a, data[session, 0][trial])

# Visualize the filtering result
for filter_type, filtered_signal, label in [
    ('Butterworth', butter_filtered, 'Butterworth Filter'),
    ('Chebyshev I', cheby1_filtered, 'Chebyshev I Filter'),
    ('Chebyshev II', cheby2_filtered, 'Chebyshev II Filter')
    ]:
    original_signal = data[session_idx, 0][trial_idx]
    filtered_signal = filtered_signal[session_idx, trial_idx]

    plot_time_and_frequency_response(original_signal, filtered_signal, fs, title=label, save_path=f'./result/question 2/{label}_session_{session_idx+1}_trial_{trial_idx+1}_frequency_response.png')
# <Mean of all trials>
plot_time_and_frequency_response_mean(data, butter_filtered, fs, num_sessions, num_trials, data_samples, label="Butterworth Filter")
plot_time_and_frequency_response_mean(data, cheby1_filtered, fs, num_sessions, num_trials, data_samples, label="Chebyshev I Filter")
plot_time_and_frequency_response_mean(data, cheby2_filtered, fs, num_sessions, num_trials, data_samples, label="Chebyshev II Filter")

# Best results among low-pass filtered data
original_signal = data[session_idx, 0][trial_idx]

butter_signal = butter_filtered[session_idx, trial_idx]
cheby1_signal = cheby1_filtered[session_idx, trial_idx]
cheby2_signal = cheby2_filtered[session_idx, trial_idx]
custom_signal = custom_filtered[session_idx, trial_idx]

plot_all_filters(original_signal, butter_signal, cheby1_signal, cheby2_signal, custom_signal, fs, save_path=f'./result/question 2/All Filter_session_{session_idx+1}_trial_{trial_idx+1}_comparison.png')
# <Mean of all trials> 
plot_all_filters_mean(data, butter_filtered, cheby1_filtered, cheby2_filtered, custom_filtered, fs, num_sessions, num_trials, data_samples)

filtered_low_tones, filtered_high_tones = separate_tone_data(data, custom_filtered, num_sessions)

plot_mean_with_standard_error(filtered_low_tones, filtered_high_tones, num_sessions, data_samples, save_path='./result/question 2/Custom Filter_mean_with_std_error.png')

# Define parameters
wind = 256          # Hanning window size
overl = [1, wind // 2, wind - 1]  # Shift sizes (1, wind/2, wind-1)
nfft = 256

# Set the color axis limits
clim_low, clim_high = -88.8, 6.2  # Lower and Upper limit for color axis

plot_spectrogram_of_mean_data(filtered_low_tones, filtered_high_tones, overl, wind, nfft, fs, max_freq, clim_low, clim_high, num_sessions)






