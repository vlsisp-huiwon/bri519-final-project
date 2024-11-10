import numpy as np

def custom_filter(data, cutoff_freq, fs):
    # Time constant (tau) for low-pass filter
    tau = 1 / (2 * np.pi * cutoff_freq)
    
    # Sampling period (inverse of sampling frequency)
    ts = 1 / fs
    
    # Initialize previous value as the first data point
    pre_val = data[0]
    
    # Create an empty array to store filtered data
    filtered_data = np.zeros_like(data)
    filtered_data[0] = pre_val  # Set the first point to the same value
    
    # Apply the low-pass filter across the data points
    for i in range(1, len(data)):
        filtered_val = (ts * data[i] + tau * pre_val) / (tau + ts)
        filtered_data[i] = filtered_val
        pre_val = filtered_val  # Update previous value for the next iteration
    
    return filtered_data


def filtfilter(data, num_sessions, num_trials, data_samples, cutoff_freq, fs):
    
    filtered_data = np.zeros((num_sessions, num_trials, data_samples))

    for session in range(num_sessions):
        for trial in range(num_trials):
            filtered_data[session, trial, :] = custom_filter(data[session, 0][trial], cutoff_freq, fs)
    
    return filtered_data