# Fourier transform - notch reject using explicit loops, display with matplotlib (plots)
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# take input
img_input = cv2.imread('pnois2.jpg', 0)
if img_input is None:
    raise FileNotFoundError("Could not read 'pnois2.jpg'")
img = img_input.copy()
h, w = img.shape
image_size = h * w


# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
#ft_shift = ft
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift) + 1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- Build ideal notch mask using exact loop format requested ---
notch_mask = np.ones((h, w), dtype=np.float32)

# user-specified notch centers (x,y) as in your earlier code and radius D
centers = [(261, 261)]   # (x, y) = (row, col) indices from spectrum
D = 1  # notch radius (adjust if needed)

for (x, y) in centers:
    for i in range(h):
        for j in range(w):
            d = math.sqrt((i - x) ** 2 + (j - y) ** 2)
            dsym = math.sqrt((i - (h - 1 - x)) ** 2 + (j - (w - 1 - y)) ** 2)
            if (d <= D) or (dsym <= D):
                notch_mask[i][j] = 0.0

# Visualize notch mask (0 -> black, 1 -> white) as uint8 for plotting
notch_mask_disp = (notch_mask * 255).astype(np.uint8)

#Apply filter here to the complex DFT (not just magnitude)
ft_shift_filtered = ft_shift * notch_mask

# Show filtered spectrum (log) - same style as original magnitude
mag_spec_filtered = 20 * np.log(np.abs(ft_shift_filtered) + 1)
mag_spec_filtered = cv2.normalize(mag_spec_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# inverse fourier (from filtered complex spectrum)
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(ft_shift_filtered)))
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- Plot results with matplotlib (2x3) ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(img_input, cmap='gray')
axs[0, 0].set_title("Input")
axs[0, 0].axis('off')

axs[0, 1].imshow(magnitude_spectrum, cmap='gray')
axs[0, 1].set_title("Magnitude Spectrum")
axs[0, 1].axis('off')

axs[0, 2].imshow(notch_mask_disp, cmap='gray')
axs[0, 2].set_title("Notch Mask (H)")
axs[0, 2].axis('off')

axs[1, 0].imshow(ang_, cmap='gray')
axs[1, 0].set_title("Phase (scaled)")
axs[1, 0].axis('off')

axs[1, 1].imshow(mag_spec_filtered, cmap='gray')
axs[1, 1].set_title("Filtered Spectrum")
axs[1, 1].axis('off')

axs[1, 2].imshow(img_back_scaled, cmap='gray')
axs[1, 2].set_title("Inverse transform")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
