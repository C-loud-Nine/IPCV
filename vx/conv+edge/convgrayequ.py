import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# -----------------------------
# Gaussian Smoothing Kernel
# -----------------------------
def gaussian_smoothing_kernel(size, sigma) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Size must be odd.")
    
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # normalize
    return kernel

# -----------------------------
# Laplacian of Gaussian (LoG) Kernel
# -----------------------------
def sharpening_kernel(size, sigma) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Size must be odd.")
    
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    r2 = x**2 + y**2

    kernel = ((r2 - 2 * sigma**2) / (sigma**4)) * np.exp(-r2 / (2 * sigma**2))
    return kernel  # zero-sum, no normalization

def apply_convolution1(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='symmetric')

    result = np.zeros_like(image, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            acc = 0.0
            for m in range(kh):
                for n in range(kw):
                    # Manual flip: kernel[kh-1-m, kw-1-n]
                    val = padded[y + m, x + n]
                    k_val = kernel[kh - 1 - m, kw - 1 - n]
                    acc += val * k_val
            result[y, x] = acc

    return {
        "raw_result": result,
        "normalized": np.round(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    }


# -----------------------------
# Parameters & Kernels
# -----------------------------
sigma = 1.0
ksize_smooth = 5   # 5σ → 5
ksize_sharp  = 7   # 7σ → 7

g_kernel  = gaussian_smoothing_kernel(ksize_smooth, sigma)
log_kernel = sharpening_kernel(ksize_sharp, sigma)

# -----------------------------
# Load Image (grayscale)
# -----------------------------
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# -----------------------------
# Apply Convolutions
# -----------------------------
#smoothed    = convolve2d(img, g_kernel, mode='same', boundary='symm')
#log_response = convolve2d(img, log_kernel, mode='same', boundary='symm')

smoothed = apply_convolution1(img, g_kernel)
log_response = apply_convolution1(img, log_kernel)

smoothed = smoothed["raw_result"]
log_response = log_response["raw_result"]


sharpened = img - log_response

# -----------------------------
# Clip for Display
# -----------------------------
smoothed   = np.clip(smoothed, 0, 255).astype(np.uint8)
sharpened  = np.clip(sharpened, 0, 255).astype(np.uint8)
log_disp   = cv2.normalize(log_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# -----------------------------
# Show Results
# -----------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,4,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis("off")
plt.subplot(1,4,2); plt.imshow(smoothed, cmap='gray'); plt.title("Gaussian Smoothing"); plt.axis("off")
plt.subplot(1,4,3); plt.imshow(log_disp, cmap='gray'); plt.title("LoG Response"); plt.axis("off")
plt.subplot(1,4,4); plt.imshow(sharpened, cmap='gray'); plt.title("Sharpened (LoG)"); plt.axis("off")
plt.show()
