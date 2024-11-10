import numpy as np
import matplotlib.pyplot as plt
from ..analysis import *
from ...fft.spectrogram import *


# Visualize the filtering result
def plot_time_and_frequency_response(original_data, filtered_data, fs, title, save_path=None):

    plt.figure(figsize=(12, 10))

    # time-domain; original vs. filtered
    plt.subplot(3, 1, 1)
    plt.plot(original_data, label='Original Data', color='black', linestyle='--')
    plt.plot(filtered_data, label='Filtered Data', color='orange')
    plt.title(f'Time-Domain Signal ({title})')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # freq.-domain: frequency response using fft comptuation - original vs. filtered
    freq_original, fft_original = compute_magnitude(original_data, fs)
    freq_filtered, fft_filtered = compute_magnitude(filtered_data, fs)

    plt.subplot(3, 1, 2)
    plt.plot(freq_original, fft_original, label='Original FFT', color='black', linestyle='--')
    plt.plot(freq_filtered, fft_filtered, label='Filtered FFT', color='orange')
    plt.title(f'Frequency-Domain Signal ({title})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.tight_layout()
    
    # ============================== rev(Here) ==============================
    # Adding the power(dB) of the signal in the freq. domain
    # Because it easily presents fitlering results.
    original_data_fft_dB = 10*np.log10((np.abs(fft_original))**2)
    filtered_data_fft_dB = 10*np.log10((np.abs(fft_filtered))**2)

    plt.subplot(3, 1, 3)
    plt.plot(freq_original, original_data_fft_dB, label='Original FFT', color='black', linestyle='--')
    plt.plot(freq_filtered, filtered_data_fft_dB, label='Filtered FFT', color='orange')
    plt.title(f'Frequency-Domain Signal^2 ({title})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power of singal (dB)')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    # plt.show()
    plt.close() 
    # =======================================================================


def plot_time_and_frequency_response_mean(data, filtered_signal, fs, num_sessions, num_trials, data_samples, label, save_path=None):
    # <Mean of all trials>
    # Initialize arrays to hold the means
    mean_original_signals = np.zeros((num_sessions, data_samples))  # Adjust to your signal length
    mean_filtered_signals = np.zeros((num_sessions, data_samples))  # Adjust to your signal length

    # Compute the mean original and filtered signals across trials for each session
    for i in range(num_sessions):
        original_signals_array = np.array([data[i, 0][j] for j in range(num_trials)])
        filtered_signals_array = np.array([filtered_signal[i][j] for j in range(num_trials)])
        
        mean_original_signals[i, :] = np.mean(original_signals_array, axis=0)
        mean_filtered_signals[i, :] = np.mean(filtered_signals_array, axis=0)

    # Plot the mean results for each session
    for session_idx in range(num_sessions):
        session_save_path = save_path if save_path else f'./result/question 2/{label}_session_{session_idx + 1}_all_trials_frequency_response.png'
        plot_time_and_frequency_response( 
            mean_original_signals[session_idx], 
            mean_filtered_signals[session_idx], 
            fs, 
            title=label + f" - Session {session_idx + 1} Mean Signals",
            save_path=session_save_path
        )


def plot_all_filters(original_data, butter_data, cheby1_data, cheby2_data, custom_data, fs, save_path=None):

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(original_data, label='Original Data', color='black', linestyle='--')
    plt.plot(butter_data, label='Butterworth Filter', color='blue')
    plt.plot(cheby1_data, label='Chebyshev I Filter', color='green')
    plt.plot(cheby2_data, label='Chebyshev II Filter', color='red')
    plt.plot(custom_data, label='Custom Low-Level Filter', color='orange')
    plt.title('Time-Domain Signal Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    freq_original, fft_original = compute_magnitude(original_data, fs)
    freq_butter, fft_butter = compute_magnitude(butter_data, fs)
    freq_cheby1, fft_cheby1 = compute_magnitude(cheby1_data, fs)
    freq_cheby2, fft_cheby2 = compute_magnitude(cheby2_data, fs)
    freq_custom, fft_custom = compute_magnitude(custom_data, fs)

    plt.subplot(3, 1, 2)
    plt.plot(freq_original, fft_original, label='Original FFT', color='black', linestyle='--')
    plt.plot(freq_butter, fft_butter, label='Butterworth FFT', color='blue')
    plt.plot(freq_cheby1, fft_cheby1, label='Chebyshev I FFT', color='green')
    plt.plot(freq_cheby2, fft_cheby2, label='Chebyshev II FFT', color='red')
    plt.plot(freq_custom, fft_custom, label='Custom Low-Level FFT', color='orange')
    plt.title('Frequency-Domain Signal Comparison (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    # ============================== rev(Here) ==============================
    # Adding the power(dB) of the signal in the freq. domain
    # Because it easily presents fitlering results.
    original_data_fft_dB = 10*np.log10((np.abs(fft_original))**2)
    butter_data_fft_dB = 10*np.log10((np.abs(fft_butter))**2)
    cheby1_data_fft_dB = 10*np.log10((np.abs(fft_cheby1))**2)
    cheby2_data_fft_dB = 10*np.log10((np.abs(fft_cheby2))**2)
    custom_data_fft_dB = 10*np.log10((np.abs(fft_custom))**2)

    plt.subplot(3, 1, 3)
    plt.plot(freq_original, original_data_fft_dB, label='Original FFT', color='black', linestyle='--')
    plt.plot(freq_butter, butter_data_fft_dB, label='Butterworth FFT', color='blue')
    plt.plot(freq_cheby1, cheby1_data_fft_dB, label='Chebyshev I FFT', color='green')
    plt.plot(freq_cheby2, cheby2_data_fft_dB, label='Chebyshev II FFT', color='red')
    plt.plot(freq_custom, custom_data_fft_dB, label='Filtered FFT', color='orange')
    plt.title(f'Frequency-Domain Signal^2 (10log10(abs.FFT))')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power of singal (dB)')
    plt.legend()
    # =======================================================================

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    plt.close() 

# <Mean of all trials> 
def plot_all_filters_mean(data, butter_filtered, cheby1_filtered, cheby2_filtered, custom_filtered, fs, num_sessions, num_trials, data_samples, save_path=None):
    # Initialize arrays to hold the means for the filtered signals
    mean_butter_filtered = np.zeros((num_sessions, data_samples))
    mean_cheby1_filtered = np.zeros((num_sessions, data_samples))
    mean_cheby2_filtered = np.zeros((num_sessions, data_samples))
    mean_custom_filtered = np.zeros((num_sessions, data_samples))

    mean_butter_filtered = np.zeros((num_sessions, data_samples))
    mean_cheby1_filtered = np.zeros((num_sessions, data_samples))
    mean_cheby2_filtered = np.zeros((num_sessions, data_samples))
    mean_custom_filtered = np.zeros((num_sessions, data_samples))

    # Compute the mean filtered signals across trials for each session
    for session in range(num_sessions):
        mean_butter_filtered[session, :] = np.mean(butter_filtered[session], axis=0)
        mean_cheby1_filtered[session, :] = np.mean(cheby1_filtered[session], axis=0)
        mean_cheby2_filtered[session, :] = np.mean(cheby2_filtered[session], axis=0)
        # mean_custom_filtered[session, :] = np.mean([low_pass_filter(data[session, 0][trial], cutoff_freq, fs) for trial in range(num_trials)], axis=0)
        mean_custom_filtered[session, :] = np.mean(custom_filtered[session], axis=0)

    # Session 1
    session_idx = 0  

    original_signal = np.mean([data[session_idx, 0][trial] for trial in range(num_trials)], axis=0)

    butter_signal = mean_butter_filtered[session_idx]
    cheby1_signal = mean_cheby1_filtered[session_idx]
    cheby2_signal = mean_cheby2_filtered[session_idx]
    custom_signal = mean_custom_filtered[session_idx]

    # Plot all filters for the session
    save_path = save_path if save_path else f'./result/question 2/All Filter_session_{session_idx + 1}_all_trials_comparison.png'
    plot_all_filters(original_signal, butter_signal, cheby1_signal, cheby2_signal, custom_signal, fs, save_path=save_path)


def plot_mean_with_standard_error(filtered_low_tones, filtered_high_tones, num_sessions, data_samples, save_path=None):
    # Initialize arrays to hold the means
    mean_low_tones = np.zeros((num_sessions, data_samples))  
    mean_high_tones = np.zeros((num_sessions, data_samples))

    for i in range(num_sessions):
        mean_low_tones[i, :] = np.mean(filtered_low_tones[i], axis=0)  
        mean_high_tones[i, :] = np.mean(filtered_high_tones[i], axis=0)

    # Calculate the overall minimum and maximum values
    minimin = np.min(np.concatenate((mean_low_tones.flatten(), mean_high_tones.flatten())))  # Overall lowest value
    maximax = np.max(np.concatenate((mean_low_tones.flatten(), mean_high_tones.flatten())))  # Overall highest value

    # Visualize low-and high-tone data for all sessions
    # num_sessions = len(filtered_low_tones)
    plt.figure(figsize=(15, num_sessions * 4))

    for session in range(num_sessions):
        ## Visualize low-tone data
        mean_low, std_error_low = mean_with_std_error(filtered_low_tones[session])
        plt.subplot(num_sessions, 2, session * 2 + 1)
        plt.fill_between(np.arange(len(mean_low)), mean_low - std_error_low, mean_low + std_error_low, color='blue', alpha=0.3)
        plt.plot(mean_low, color='blue', label='Mean Low Tone')
        plt.title(f'Session {session + 1}: Low-tone Response')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.ylim([minimin*1.1, maximax*1.1])
        plt.legend()
        plt.grid(True)

        # Visualize high-tone data
        mean_high, std_error_high = mean_with_std_error(filtered_high_tones[session])
        plt.subplot(num_sessions, 2, session * 2 + 2)
        plt.fill_between(np.arange(len(mean_high)), mean_high - std_error_high, mean_high + std_error_high, color='red', alpha=0.3)
        plt.plot(mean_high, color='red', label='Mean High Tone')
        plt.title(f'Session {session + 1}: High-tone Response')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.ylim([minimin*1.1, maximax*1.1])
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    plt.close() 


def plot_spectrogram_of_mean_data(filtered_low_tones, filtered_high_tones, overl, wind, nfft, fs, max_freq, clim_low, clim_high, num_sessions, save_path=None):

    for session in range(num_sessions):
        session_save_path = save_path if save_path else f'./result/question 2/Custom Filter_session_{session + 1}_Spectrogram.png'
        plt.figure(figsize=(18, 10))

        for jj, overlap in enumerate(overl):
            # Plot spectrograms for low-tone response
            plt.subplot(2, len(overl), jj + 1)
            mean_low = np.mean(filtered_low_tones[session], axis=0)
            f, tt, Sxx = custom_spectrogram(mean_low, fs, wind, overlap, nfft)
            
            # Convert dB scale: just define variable
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)

            img = plt.pcolormesh(tt * 1000, f / 1000, Sxx_dB, cmap='jet')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.title(f'Session {session + 1} Low-tone Spectrogram (Overlap={overlap})')
            plt.ylabel('Frequency (kHz)')
            plt.xlabel('Time (ms)')
            plt.ylim([0, max_freq / 1000])

            # Set color limits to clip the data
            img.set_clim(clim_low, clim_high)

            # Plot spectrograms for high-tone response
            plt.subplot(2, len(overl), len(overl) + jj + 1)
            mean_high = np.mean(filtered_high_tones[session], axis=0)
            f, tt, Sxx = custom_spectrogram(mean_high, fs, wind, overlap, nfft)
            
            # Convert dB scale: just define variable
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)

            img = plt.pcolormesh(tt * 1000, f / 1000, Sxx_dB, cmap='jet')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.title(f'Session {session + 1} High-tone Spectrogram (Overlap={overlap})')
            plt.ylabel('Frequency (kHz)')
            plt.xlabel('Time (ms)')
            plt.ylim([0, max_freq / 1000])

            # Set color limits to clip the data
            img.set_clim(clim_low, clim_high)

        plt.tight_layout()
        # plt.show
        plt.savefig(session_save_path)
        plt.close() 