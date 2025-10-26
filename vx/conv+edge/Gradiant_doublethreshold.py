import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Generate 1st order derivative Gaussian kernels
# -----------------------------
def gaussian_derivative_kernels(sigma):
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1  # ensure odd size
    
    center = size // 2
    x = np.arange(-center, center + 1, 1)
    
    # 1D Gaussian derivative
    gx_1d = -x * np.exp(-x**2 / (2 * sigma**2))
    gy_1d = gx_1d.copy()
    
    # 2D kernels
    Gx = np.outer(gx_1d, np.exp(-x**2 / (2 * sigma**2)))
    Gy = np.outer(np.exp(-x**2 / (2 * sigma**2)), gy_1d)
    
    # Normalize
    Gx /= np.sum(np.abs(Gx))
    Gy /= np.sum(np.abs(Gy))
    
    return Gx.astype(np.float32), Gy.astype(np.float32)

# -----------------------------
# Double thresholding function
# -----------------------------
def double_threshold(img, T_low, T_high):
    strong = 255
    weak = 128
    
    res = np.zeros_like(img, dtype=np.uint8)
    res[img >= T_high] = strong
    res[(img >= T_low) & (img < T_high)] = weak
    return res

# -----------------------------
# Parameters
# -----------------------------
sigma = 1.0
T_low = 50
T_high = 100

# -----------------------------
# Load grayscale image
# -----------------------------
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found: lena.jpg")
img = img.astype(np.float32)

# -----------------------------
# Generate derivative kernels
# -----------------------------
kx, ky = gaussian_derivative_kernels(sigma)

# -----------------------------
# Apply convolution with derivative kernels
# -----------------------------
gx = cv2.filter2D(img, -1, kx)
gy = cv2.filter2D(img, -1, ky)

# -----------------------------
# Gradient magnitude
# -----------------------------
grad_mag = cv2.magnitude(gx, gy)
grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# -----------------------------
# Double thresholding
# -----------------------------
edges = double_threshold(grad_mag_norm, T_low, T_high)

# -----------------------------
# Display results
# -----------------------------
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Gradient & Double Thresholding Edge Detection", fontsize=16)

axs[0,0].imshow(img, cmap='gray'); axs[0,0].set_title("Original"); axs[0,0].axis('off')
axs[0,1].imshow(gx, cmap='gray'); axs[0,1].set_title("Gradient X"); axs[0,1].axis('off')
axs[0,2].imshow(gy, cmap='gray'); axs[0,2].set_title("Gradient Y"); axs[0,2].axis('off')
axs[1,0].imshow(grad_mag, cmap='gray'); axs[1,0].set_title("Gradient Magnitude"); axs[1,0].axis('off')
axs[1,1].imshow(grad_mag_norm, cmap='gray'); axs[1,1].set_title("Normalized Gradient"); axs[1,1].axis('off')
axs[1,2].imshow(edges, cmap='gray'); axs[1,2].set_title("Double Threshold Edges"); axs[1,2].axis('off')

plt.tight_layout()
plt.show()
