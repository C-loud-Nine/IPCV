import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from copy import deepcopy as dpc

# take input
img_input = cv2.imread('input.jpg', 0)
if img_input is None:
    raise FileNotFoundError("Could not read 'input.jpg'")
img = dpc(img_input)

h, w = img.shape

# Fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

# Magnitude and phase spectra (use cv2.normalize like in your previous code)
magnitude_spectrum = np.log(np.abs(ft_shift) + 1)
magnitude_spectrum_scaled = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
phase = np.angle(ft_shift)
phase_scaled = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Build TRUE 2D notch filter (mask) the same way as you did, but to apply to complex DFT!
notch_mask = np.ones((h, w), dtype=np.float32)

# Example notch location/size (change as needed for your pattern)
x, y = 112, 112
D = 20

for i in range(h):
    for j in range(w):
        d = math.sqrt((i - x) ** 2 + (j - y) ** 2)
        dsym = math.sqrt((i - (h - 1 - x)) ** 2 + (j - (w - 1 - y)) ** 2)
        if (d <= D) or (dsym <= D):
            notch_mask[i][j] = 0.0

# Visualize notch mask (white = keep, black = zeroed out) using cv2.normalize
notch_mask_disp = cv2.normalize(notch_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Apply notch filter IN FREQUENCY DOMAIN
ft_shift_filtered = ft_shift * notch_mask

# Show filtered spectrum (use cv2.normalize like before)
mag_spec_filtered = np.log(np.abs(ft_shift_filtered) + 1)
mag_spec_filtered_scaled = cv2.normalize(mag_spec_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Inverse Fourier transform to spatial domain
ft_unshift = np.fft.ifftshift(ft_shift_filtered)
img_back_complex = np.fft.ifft2(ft_unshift)
img_back = np.real(img_back_complex)
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- Plot everything using matplotlib ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(img_input, cmap='gray')
axs[0, 0].set_title("Input")
axs[0, 0].axis('off')

axs[0, 1].imshow(magnitude_spectrum_scaled, cmap='gray')
axs[0, 1].set_title("Magnitude Spectrum")
axs[0, 1].axis('off')

axs[0, 2].imshow(notch_mask_disp, cmap='gray')
axs[0, 2].set_title("Notch Mask")
axs[0, 2].axis('off')

axs[1, 0].imshow(phase_scaled, cmap='gray')
axs[1, 0].set_title("Phase")
axs[1, 0].axis('off')

axs[1, 1].imshow(mag_spec_filtered_scaled, cmap='gray')
axs[1, 1].set_title("Filtered Spectrum")
axs[1, 1].axis('off')

axs[1, 2].imshow(img_back_scaled, cmap='gray')
axs[1, 2].set_title("Inverse Transform")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()


