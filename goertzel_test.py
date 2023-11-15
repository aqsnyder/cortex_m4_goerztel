# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bessel, lfilter, welch

# %% [markdown]
# Goertzel reference: <br> https://www.embedded.com/the-goertzel-algorithm/ <br>
# https://www.dsp-weimich.com/digital-signal-processing/goertzel-algorithm-and-c-implementation-using-the-octave-gnu-tool/

# %%
# Goertzel algorithm function
def goertzel(samples, sample_rate, target_freq, n):
    """
    The Goertzel algorithm is used to calculate the magnitude of a specific frequency component within a signal.
    This is particularly efficient for computing the spectral content at specific frequencies of interest.
    """
    k = int(0.5 + n * target_freq / sample_rate)
    omega = (2 * np.pi / n) * k
    coeff = 2 * np.cos(omega)
    q0, q1, q2 = 0, 0, 0
    for sample in samples:
        q0 = coeff * q1 - q2 + sample
        q2 = q1
        q1 = q0
    real = (q1 - q2 * np.cos(omega))
    imag = (q2 * np.sin(omega))
    return np.sqrt(real**2 + imag**2)

# Function to calculate magnitude response over a range of frequencies
def magnitude_response(signal, sample_rate, freq_range, n):
    magnitudes = []
    for freq in freq_range:
        magnitude = goertzel(signal, sample_rate, freq, n)
        magnitudes.append(magnitude)
    return magnitudes

# Function to calculate power response over a range of frequencies
def power_response(signal, sample_rate, freq_range, n):
    powers = []
    for freq in freq_range:
        magnitude = goertzel(signal, sample_rate, freq, n)
        power = magnitude**2  # Squaring the magnitude to get power
        normalized_power = power / n**2  # Normalization by the square of the number of samples
        powers.append(normalized_power)
    return powers

# Function to apply a 4th order Bessel filter
def apply_bessel_filter(signal, sample_rate, cutoff_freq):
    """
    This function applies a 4th order Bessel filter to the signal.
    The Bessel filter is known for its linear phase response but has a gentler roll-off compared to other filters.
    The cutoff frequency determines the frequency point at which the filter starts attenuating the signal.
    """
    b, a = bessel(N=4, Wn=cutoff_freq/(0.5*sample_rate), btype='low', analog=False)
    return lfilter(b, a, signal)

def calculate_psd_rfu(signal, sample_rate, center_freq, bandwidth):
    """
    This function calculates the Power Spectral Density (PSD) which will be used as RFU's.
    PSD provides a measure of the signal's power content versus frequency.
    RFU is calculated as the sum of PSD values within a specified frequency range, indicating the signal's strength in that range.
    """

    """
    The function uses Welch's method (via the welch function from libraries like SciPy)
    to estimate the PSD of the input signal. Welch's method divides the signal
    into overlapping segments, computes a modified periodogram for each segment, and then averages
    these periodograms to estimate the PSD
    """
    f, Pxx = welch(signal, fs=sample_rate, nperseg=1024)   # nperseg is the number of samples per segment (decrease for less noise)
    # Find the frequency range of interest
    freq_range = (f >= center_freq - bandwidth/2) & (f <= center_freq + bandwidth/2)
    # Isolate the PSD values within the frequency range of interest
    Pxx_interest = Pxx[freq_range]
    f_interest = f[freq_range]
    # Calculate RFU as the sum of PSD in the frequency range of interest
    rfu = np.sum(Pxx_interest)
    return f_interest, Pxx_interest, rfu

# Function to add white Gaussian noise to a signal
def add_white_noise(signal, noise_level):
    """
    This function adds white Gaussian noise to the signal.
    The noise level is a fraction of the signal's standard deviation, allowing control over the signal-to-noise ratio.
    Adding noise can be useful for testing the robustness of signal processing algorithms.
    """
    mean_noise = 0
    std_noise = noise_level * np.std(signal)
    noise = np.random.normal(mean_noise, std_noise, len(signal))
    return signal + noise
