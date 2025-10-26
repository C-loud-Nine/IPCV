import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Gaussian Derivative Kernels ===
def gaussian_derivative_kernels(sigma):
    size = int(np.ceil(7 * sigma))
    if size % 2 == 0:
        size += 1
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    gaussian = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x**2 + y**2) / (2 * sigma ** 2))
    gx = -x / (sigma ** 2) * gaussian
    gy = -y / (sigma ** 2) * gaussian
    return gx, gy

# === Convolution ===
def apply_convolution(image, kernel):
    return cv2.filter2D(image, cv2.CV_32F, kernel)

# === Hysteresis Thresholding ===
def hysteresis_threshold(magnitude, T_low, T_high):
    strong = 255
    weak = 128

    edges = np.zeros_like(magnitude, dtype=np.uint8)

    # Step 1: classify pixels
    strong_i, strong_j = np.where(magnitude >= T_high)
    weak_i, weak_j = np.where((magnitude >= T_low) & (magnitude < T_high))

    edges[strong_i, strong_j] = strong
    edges[weak_i, weak_j] = weak

    # Step 2: edge tracking by hysteresis
    M, N = edges.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if edges[i, j] == weak:
                # If at least one neighbor is strong → promote to strong
                if np.any(edges[i-1:i+2, j-1:j+2] == strong):
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0
    return edges

# === Workflow ===
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

sigma = 1.0
gx_kernel, gy_kernel = gaussian_derivative_kernels(sigma)

grad_x = apply_convolution(img, gx_kernel)
grad_y = apply_convolution(img, gy_kernel)

magnitude = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))

# Thresholds
T_low, T_high = 10, 20

edges_hyst = hysteresis_threshold(magnitude, T_low, T_high)

# Show results (compare magnitude, double threshold, hysteresis)
plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gx_kernel, cmap='gray')
plt.title("Gx Kernel")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(gy_kernel, cmap='gray')
plt.title("Gy Kernel")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(grad_x, cmap='gray')
plt.title("Gradient X")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(grad_y, cmap='gray')
plt.title("Gradient Y")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(magnitude, cmap='gray')
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(edges_hyst, cmap='gray')
plt.title(f"Hysteresis (T_low={T_low}, T_high={T_high})")
plt.axis("off")

plt.tight_layout()
plt.show()
