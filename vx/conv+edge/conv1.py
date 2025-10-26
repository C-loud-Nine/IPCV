import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# Load grayscale image
# -------------------------------
img_path = 'box.jpg'  # path to your image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
h, w = img.shape  # height and width of image

# ===============================
# Task: 5x5 Convolution (anchor at (3,3))
# ===============================

# Define a 5x5 smoothing kernel
kernel = np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32)

kh, kw = kernel.shape  # kernel height and width

# -------------------------------
# Anchor (kernel center) - change here if needed
# -------------------------------
# Default symmetric center for 5x5 is (2,2). Here we set anchor to (3,3).
aH, aW = 3, 3

# Compute asymmetric padding from anchor:
pad_top = aH
pad_bottom = kh - 1 - aH
pad_left = aW
pad_right = kw - 1 - aW

print("Anchor (aH,aW):", (aH, aW))
print("padding (top,bottom,left,right) =", (pad_top, pad_bottom, pad_left, pad_right))

# -------------------------------
# Padding
# -------------------------------
# Use asymmetric padding so kernel fits at borders
img_bordered = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

# -------------------------------
# Manual convolution
# -------------------------------
conv_manual = np.zeros_like(img_bordered, dtype=np.float32)
for i in range(pad_top, pad_top + h):
    for j in range(pad_left, pad_left + w):
        result = 0.0
        for m in range(kh):
            for n in range(kw):
                # Implicit flip of kernel indices for convolution (true convolution)
                # Map kernel index (m,n) to flipped: (kh-1-m, kw-1-n)
                # Map image index relative to current padded position i,j using anchor:
                # image index = (i + m - aH, j + n - aW)
                result += kernel[kh - 1 - m, kw - 1 - n] * img_bordered[i + m - aH, j + n - aW]
        conv_manual[i, j] = result

# -------------------------------
# Normalization and cropping
# -------------------------------
# Scale float values to 0-255 for visualization
conv_norm = cv2.normalize(conv_manual, None, 0, 255, cv2.NORM_MINMAX)
# Crop the padded border to return to original image size
conv_cropped = np.round(conv_norm).astype(np.uint8)[pad_top:pad_top + h, pad_left:pad_left + w]

# Maximum value for full white visualization (before normalization)
max_val = np.sum(kernel) * 255  # all 255 pixels multiplied by kernel sum

# ===============================
# Display using matplotlib
# ===============================
fig, axs = plt.subplots(1, 5, figsize=(20,5))
fig.suptitle(f"5x5 Convolution (Anchor at ({aH},{aW}))", fontsize=16)

axs[0].imshow(img, cmap='gray'); axs[0].set_title("Original"); axs[0].axis('off')
axs[1].imshow(img_bordered, cmap='gray'); axs[1].set_title("Padded"); axs[1].axis('off')
axs[2].imshow(conv_manual, cmap='gray', vmin=0, vmax=max_val); axs[2].set_title("Convolution (float)"); axs[2].axis('off')
axs[3].imshow(conv_norm, cmap='gray'); axs[3].set_title("Normalized"); axs[3].axis('off')
axs[4].imshow(conv_cropped, cmap='gray'); axs[4].set_title("Normalized + Cropped"); axs[4].axis('off')

plt.tight_layout()
plt.show()

# ===============================
# OpenCV version (commented)
# ===============================
# Using cv2.filter2D, which assumes kernel center is (kh//2, kw//2).
# If anchor != (kh//2, kw//2) the alignment will differ.
# img_conv = cv2.filter2D(img_bordered, ddepth=cv2.CV_32F, kernel=kernel)
# norm_cv = np.round(cv2.normalize(img_conv, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
# norm_cropped_cv = norm_cv[pad_top:pad_top+h, pad_left:pad_left+w]
# 
# cv2.imshow('Original', img)
# cv2.imshow('Padded', img_bordered)
# cv2.imshow('Convolution (cv2)', img_conv)
# cv2.imshow('Normalized (cv2)', norm_cv)
# cv2.imshow('Normalized + Cropped (cv2)', norm_cropped_cv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
