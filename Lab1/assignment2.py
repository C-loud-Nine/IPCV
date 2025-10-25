import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Define Kernels
# -----------------------------
def gaussian_smoothing_kernel(size, sigma):
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

def laplacian_of_gaussian_kernel(size, sigma):
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    r2 = x**2 + y**2
    kernel = ((r2 - 2 * sigma**2) / (sigma**4)) * np.exp(-r2 / (2 * sigma**2))
    return kernel.astype(np.float32)  # zero-sum, no normalization

sigma_smooth = 1.0
sigma_sharp = 1.0
ksize_smooth = 5
ksize_sharp = 7

g_kernel = gaussian_smoothing_kernel(ksize_smooth, sigma_smooth)
log_kernel = laplacian_of_gaussian_kernel(ksize_sharp, sigma_sharp)

# -----------------------------
# Load Color Image
# -----------------------------
img_color = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
if img_color is None:
    raise FileNotFoundError("Image not found: lena.jpg")

# -----------------------------
# Apply Convolution on RGB Channels
# -----------------------------
b, g, r = cv2.split(img_color)

rgb_channels = [b, g, r]
rgb_smoothed = []
rgb_sharpened = []

for ch in rgb_channels:
    smoothed = cv2.filter2D(ch, -1, g_kernel)
    log_resp = cv2.filter2D(ch, -1, log_kernel)
    sharpened = ch - log_resp

    # Normalize for display
    smoothed = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rgb_smoothed.append(smoothed)
    rgb_sharpened.append(sharpened)

rgb_smoothed_img = cv2.merge(rgb_smoothed)
rgb_sharpened_img = cv2.merge(rgb_sharpened)

# -----------------------------
# Convert RGB to HSV
# -----------------------------
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)

hsv_channels = [h, s, v]
hsv_smoothed = []
hsv_sharpened = []

for ch in hsv_channels:
    smoothed = cv2.filter2D(ch, -1, g_kernel)
    log_resp = cv2.filter2D(ch, -1, log_kernel)
    sharpened = ch - log_resp

    # Normalize for display
    smoothed = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hsv_smoothed.append(smoothed)
    hsv_sharpened.append(sharpened)

hsv_smoothed_img = cv2.merge(hsv_smoothed)
hsv_sharpened_img = cv2.merge(hsv_sharpened)

hsv_smoothed_bgr = cv2.cvtColor(hsv_smoothed_img, cv2.COLOR_HSV2BGR)
hsv_sharpened_bgr = cv2.cvtColor(hsv_sharpened_img, cv2.COLOR_HSV2BGR)

# -----------------------------
# Display Results
# -----------------------------
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("RGB and HSV Convolution Results", fontsize=16)

# Original
axs[0,0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
axs[0,0].set_title("Original RGB"); axs[0,0].axis('off')

# RGB Smoothing & Sharpening
axs[0,1].imshow(cv2.cvtColor(rgb_smoothed_img, cv2.COLOR_BGR2RGB))
axs[0,1].set_title("RGB Gaussian Smoothing"); axs[0,1].axis('off')

axs[0,2].imshow(cv2.cvtColor(rgb_sharpened_img, cv2.COLOR_BGR2RGB))
axs[0,2].set_title("RGB LoG Sharpened"); axs[0,2].axis('off')

# Original HSV
axs[1,0].imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
axs[1,0].set_title("Original HSV"); axs[1,0].axis('off')

# HSV Smoothing & Sharpening
axs[1,1].imshow(cv2.cvtColor(hsv_smoothed_bgr, cv2.COLOR_BGR2RGB))
axs[1,1].set_title("HSV Gaussian Smoothing"); axs[1,1].axis('off')

axs[1,2].imshow(cv2.cvtColor(hsv_sharpened_bgr, cv2.COLOR_BGR2RGB))
axs[1,2].set_title("HSV LoG Sharpened"); axs[1,2].axis('off')

# Show individual channels for reference (optional)
axs[2,0].imshow(cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX), cmap='gray')
axs[2,0].set_title("Blue Channel (RGB)"); axs[2,0].axis('off')

axs[2,1].imshow(cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX), cmap='gray')
axs[2,1].set_title("H Channel (HSV)"); axs[2,1].axis('off')

axs[2,2].imshow(cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX), cmap='gray')
axs[2,2].set_title("V Channel (HSV)"); axs[2,2].axis('off')

plt.tight_layout()
plt.show()
