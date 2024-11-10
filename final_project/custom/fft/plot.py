import matplotlib.pyplot as plt
from .fft import *

def comparison_of_lib_fft(fs, time, sig, padded_signal, fft_huiwon):
    # Create a figure to visualize the results
    plt.figure(figsize=(12, 8))

    # 4. Chirp signal (time domain)
    plt.subplot(4, 1, 1)
    plt.plot(time, sig)
    plt.title('Chirp Signal (Time Domain)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    ## Calculate frequency bins up to fs/2 for the FFT result
    frequencies = custom_fftfreq(len(padded_signal), 1/fs)

    # 5. FFT (low-level implementation)
    plt.subplot(4, 1, 2)
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_huiwon[:len(frequencies)//2]))
    plt.title("FFT (Low-Level Implementation)")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # 6. Compare with numpy FFT
    fft_numpy = np.fft.fft(padded_signal)

    plt.subplot(4, 1, 3)
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_numpy[:len(frequencies)//2]))
    plt.title("FFT (Using numpy)")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # 7. Visualize difference between fft_huiwon and fft_numpy
    difference = np.abs(fft_huiwon - fft_numpy)
    plt.subplot(4, 1, 4)
    plt.plot(frequencies[:len(frequencies)//2], difference[:len(frequencies)//2])
    plt.title("Difference between huiwon FFT and numpy FFT")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude difference")

    plt.tight_layout()
    # plt.show()
    plt.savefig('./result/question 1/FFT result.png')
    plt.close() 