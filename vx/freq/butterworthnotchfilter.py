import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_notch_reject_mask(shape, centers, D0=5.0, n=2):
    """
    shape: (h, w)
    centers: list of (row, col) tuples giving notch centers in the shifted spectrum coordinates
             (i.e., coordinates as you see in magnitude_spectrum after fftshift)
    D0: cutoff radius
    n: Butterworth order
    Returns a float32 mask with values in (0,1]
    """
    h, w = shape
    # create grid of indices (row, col)
    rows = np.arange(h).reshape(h, 1)
    cols = np.arange(w).reshape(1, w)
    mask = np.ones((h, w), dtype=np.float32)

    for (r0, c0) in centers:
        # symmetric center for the shifted spectrum
        rsym = (h - 1) - r0
        csym = (w - 1) - c0

        # distance arrays (vectorized)
        d1 = np.sqrt((rows - r0)**2 + (cols - c0)**2)
        d2 = np.sqrt((rows - rsym)**2 + (cols - csym)**2)

        # Avoid division by zero: where d==0, set the response to 0 (full rejection).
        # Compute Butterworth notch reject: H = 1 / (1 + (D0 / d)^(2n))
        # For d==0 set H=0 explicitly
        with np.errstate(divide='ignore', invalid='ignore'):
            H1 = 1.0 / (1.0 + (D0 / d1)**(2.0 * n))
            H2 = 1.0 / (1.0 + (D0 / d2)**(2.0 * n))
        H1[d1 == 0] = 0.0
        H2[d2 == 0] = 0.0

        mask *= (H1 * H2)

    return mask

# --- load image ---
img_input = cv2.imread('two_noise.jpeg', 0)
if img_input is None:
    raise FileNotFoundError("Could not read 'pnois2.jpg'")

img = img_input.astype(np.float32)
h, w = img.shape

# --- forward DFT (shifted) ---
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

# magnitude & phase for display
magnitude_spectrum = 20 * np.log(np.abs(ft_shift) + 1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
phase = np.angle(ft_shift)
phase_disp = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- specify centers (row, col) in the shifted spectrum ---
# Example: remove two periodic peaks you inspected: (272,256) and (261,261)
centers = [(272, 256), (261, 261)]
D0 = 20.0
n = 2

# build mask (vectorized)
notch_mask = butterworth_notch_reject_mask((h, w), centers, D0=D0, n=n)
notch_mask_disp = (notch_mask * 255).astype(np.uint8)

# --- apply filter in frequency domain ---
ft_shift_filtered = ft_shift * notch_mask

mag_spec_filtered = 20 * np.log(np.abs(ft_shift_filtered) + 1)
mag_spec_filtered = cv2.normalize(mag_spec_filtered, None, 0, 255,
                                  cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- inverse FFT ---
img_back = np.fft.ifftshift(ft_shift_filtered)
img_back = np.fft.ifft2(img_back)
img_back = np.real(img_back)
img_back_scaled = cv2.normalize(img_back, None, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- show results ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(img_input, cmap='gray'); axs[0, 0].set_title("Input"); axs[0, 0].axis('off')
axs[0, 1].imshow(magnitude_spectrum, cmap='gray'); axs[0, 1].set_title("Magnitude Spectrum"); axs[0, 1].axis('off')
axs[0, 2].imshow(notch_mask_disp, cmap='gray'); axs[0, 2].set_title(f"Butterworth Notch Mask\nD0={D0}, n={n}"); axs[0, 2].axis('off')
axs[1, 0].imshow(phase_disp, cmap='gray'); axs[1, 0].set_title("Phase"); axs[1, 0].axis('off')
axs[1, 1].imshow(mag_spec_filtered, cmap='gray'); axs[1, 1].set_title("Filtered Spectrum"); axs[1, 1].axis('off')
axs[1, 2].imshow(img_back_scaled, cmap='gray'); axs[1, 2].set_title("Inverse Transform"); axs[1, 2].axis('off')
plt.tight_layout()
plt.show()
