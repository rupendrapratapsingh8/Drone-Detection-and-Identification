import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =====================================================================
# 1. PARAMETERS & CONFIGURATION
# =====================================================================
fs = 20e6          # 20 MHz Sampling Rate (HackRF One Max Bandwidth)
N = 1024           # Samples per discrete FFT processing block
num_blocks = 500   # Total time duration partitioned into sample blocks
pfa = 0.01         # Targeted Probability of False Alarm for CFAR 

print(f"[i] Initializing DSP Pipeline: Fs = {fs/1e6} MHz, FFT Size = {N}")

# Generate Synthetic Baseband Noise Floor (Hypothesis H0)
# Modeled as a complex Additive White Gaussian Noise (AWGN) vector
noise_power = 1.0
noise = np.random.normal(0, np.sqrt(noise_power/2), (num_blocks, N)) + \
        1j * np.random.normal(0, np.sqrt(noise_power/2), (num_blocks, N))

# =====================================================================
# 2. SIMULATING DRONE SIGNALS & FHSS BURSTS (Hypothesis H1)
# =====================================================================
# Copy noise template matrix to introduce localized drone emissions
signal_matrix = noise.copy()
snr_db = 12  # Signal-to-Noise Ratio of incoming UAV transmission
signal_power = noise_power * (10 ** (snr_db / 10))
t = np.arange(N)

# Burst A: Occurs at Time Blocks 100-150, centered at discrete bin index 300
burst1_center = 300
burst1_width = 32
for block in range(100, 150):
    frequency_offset = (burst1_center - N//2) / N
    burst_signal = np.sqrt(signal_power) * np.exp(2j * np.pi * frequency_offset * t)
    # Injecting narrow signal window over the noise floor
    signal_matrix[block, burst1_center - burst1_width//2 : burst1_center + burst1_width//2] += \
        burst_signal[0 : burst1_width]

# Burst B: Frequency Hop occurs at Time Blocks 300-350, jumping to bin index 750
burst2_center = 750
burst2_width = 32
for block in range(300, 350):
    frequency_offset = (burst2_center - N//2) / N
    burst_signal = np.sqrt(signal_power) * np.exp(2j * np.pi * frequency_offset * t)
    signal_matrix[block, burst2_center - burst2_width//2 : burst2_center + burst2_width//2] += \
        burst_signal[0 : burst2_width]

# =====================================================================
# 3. ENERGY DETECTION & ALARM DEPLOYMENT
# =====================================================================
# Equation 3 implementation: Computes total energy per block E = sum(|x(n)|^2)
block_energies = np.sum(np.abs(signal_matrix)**2, axis=1)

# Statistically modeling the H0 noise threshold via Central Limit Theorem
mu_eg = N * noise_power               # Expected value of energy metric under H0
sigma_eg = np.sqrt(N) * noise_power   # Variance of energy metric under H0

# Find inverse normal CDF value to satisfy target Probability of False Alarm
z_alpha = norm.ppf(1 - pfa)
gamma_threshold = mu_eg + z_alpha * sigma_eg

print(f"[i] Statically Stabilized CFAR Threshold Calculated: {gamma_threshold:.2f}")

# Compare calculated metric against the dynamic threshold
uav_detections = block_energies > gamma_threshold
detection_indices = np.where(uav_detections)[0]

print(f"[!] UAV Presence Flagged! Drone detected in {len(detection_indices)} processing blocks.")

# =====================================================================
# 4. SPECTROGRAM FEATURE ANALYSIS & VISUALIZATION
# =====================================================================
# Fast Fourier Transform (FFT) extraction converted to Decibel units
fft_matrix = np.fft.fftshift(np.fft.fft(signal_matrix, axis=1), axes=1)
spectrogram_db = 10 * np.log10(np.abs(fft_matrix)**2)

# Instantiate plotting window
plt.figure(figsize=(12, 8))

# Top Graph: RF Waterfall Display (Spectrogram View)
plt.subplot(2, 1, 1)
frequencies_mhz = np.linspace(-fs/2, fs/2, N) / 1e6
plt.imshow(spectrogram_db, aspect='auto', extent=[frequencies_mhz[0], frequencies_mhz[-1], num_blocks, 0], cmap='plasma')
plt.colorbar(label='Power Spectral Density (dB)')
plt.title('HackRF One Simulated Output: Passive RF Spectrogram Window')
plt.ylabel('Discrete Processing Time Blocks')
plt.xlabel('Frequency Offset from Carrier Center (MHz)')

# Bottom Graph: Decision Engine Evaluation 
plt.subplot(2, 1, 2)
plt.plot(block_energies, label='Computed Signal Energy ($E$)', color='midnightblue', linewidth=1.5)
plt.axhline(y=gamma_threshold, color='crimson', linestyle='--', linewidth=2, label=f'CFAR Threshold ($\gamma$, Pfa={pfa})')
plt.fill_between(range(num_blocks), block_energies, gamma_threshold, where=(block_energies > gamma_threshold), 
                 color='orangered', alpha=0.3, label='Active UAV Detection Event')
plt.title('Real-Time Energy Detector Decision Output Matrix')
plt.xlabel('Discrete Processing Time Blocks')
plt.ylabel('Calculated Signal Energy Vector')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('drone_detection_dsp_output.png', dpi=300)
print("[+] Graphical matrix output exported successfully as 'drone_detection_dsp_output.png'.")
plt.show()
