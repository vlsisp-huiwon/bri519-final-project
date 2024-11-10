import numpy as np

# ============================== rev(Here) ==============================
# def huiwon_fft(x):
#     N = len(x)
#     if N <= 1:
#         return x
    
#     # Ensure the input length is even 
#     # if not, pad with a zero
#     # This avoids issues in the recursive splitting process
#     if N % 2 != 0:
#         x = np.pad(x, (0, 1), mode='constant')
    
#     even = huiwon_fft(x[0::2])
#     odd = huiwon_fft(x[1::2])
#     T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
#     return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]
def huiwon_fft(x):
    N = len(x)
    if N <= 1:
        return x
    
    # Ensure the input length is even 
    # if not, pad with a zero
    # This avoids issues in the recursive splitting process
    if N % 2 != 0:
        x = np.append(x, 0)
        N += 1
    
    even = huiwon_fft(x[0::2])
    odd = huiwon_fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]
# =======================================================================

# ============================== rev-add(Here) ==============================
def custom_fftfreq(n, d):
    freqs = []
    
    for i in range(0, n //2 + 1):
        freqs.append(i / (n*d))
    
    for i in range(-n // 2, 0):
        freqs. append(i / (n*d))
    
    return freqs

def custom_rfftfreq(n, d):
    val = 1.0 / (n * d)
    return np.array([k * val for k in range(n // 2 + 1)])

def custom_rfft(x, nfft):

    fft_result = huiwon_fft(x)
    
    # If the result is shorter than nfft, fft_result is padded with "0".
    if len(fft_result) < nfft:
        # fft_result = np.pad(fft_result, (0, nfft - len(fft_result)), 'constant', constant_values=np.mean(x))
        fft_result = np.pad(fft_result, (0, nfft - len(fft_result)), 'constant')
   
    return fft_result[:nfft // 2 + 1]     # If the result is longer than nfft, fft_result is cropped.
# =======================================================================

