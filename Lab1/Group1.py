import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# Load grayscale image
# -------------------------------
img_path = 'box.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
h, w = img.shape

# ===============================
# Task 1: 5x5 Convolution (center at 2,2)
# ===============================
kernel1 = np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32)

kh, kw = kernel1.shape
pad_h, pad_w = 2, 2  # center at (2,2)

# Add border
img_bordered = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)
conv_manual = np.zeros_like(img_bordered, dtype=np.float32)

# Manual convolution using loops
for i in range(pad_h, h + pad_h):
    for j in range(pad_w, w + pad_w):
        result = 0.0
        for m in range(kh):
            for n in range(kw):
                result += kernel1[kh - 1 - m, kw - 1 - n] * img_bordered[i - pad_h + m, j - pad_w + n]
        conv_manual[i, j] = result

# Normalize and crop
conv_norm = cv2.normalize(conv_manual, None, 0, 255, cv2.NORM_MINMAX)
conv_cropped = np.round(conv_norm).astype(np.uint8)[pad_h:h+pad_h, pad_w:w+pad_w]

# Plot Task 1
max_val = np.sum(kernel1) * 255  # for full white display before normalization
fig, axs = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("Task 1: 5x5 Convolution (Center at 2,2)", fontsize=16)
axs[0].imshow(img, cmap='gray'); axs[0].set_title("Original"); axs[0].axis('off')
axs[1].imshow(img_bordered, cmap='gray'); axs[1].set_title("Padded"); axs[1].axis('off')
axs[2].imshow(conv_manual, cmap='gray', vmin=0, vmax=max_val); axs[2].set_title("Convolution (float, full white)"); axs[2].axis('off')
axs[3].imshow(conv_norm, cmap='gray'); axs[3].set_title("Normalized"); axs[3].axis('off')
axs[4].imshow(conv_cropped, cmap='gray'); axs[4].set_title("Normalized + Cropped"); axs[4].axis('off')
plt.tight_layout()
plt.show()


# ===============================
# Task 2: Prewitt Operator
# ===============================
kernel_px = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
kernel_py = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]], dtype=np.float32)
kh2, kw2 = 3, 3

# ----------- Anchor at (0,0) -----------
pad_top0, pad_left0 = 0, 0
pad_bottom0, pad_right0 = kh2 - 1, kw2 - 1
img_b0 = cv2.copyMakeBorder(img, pad_top0, pad_bottom0, pad_left0, pad_right0, cv2.BORDER_CONSTANT)
gx0 = np.zeros_like(img_b0, dtype=np.float32)
gy0 = np.zeros_like(img_b0, dtype=np.float32)

for i in range(h):
    for j in range(w):
        sum_x = 0.0
        sum_y = 0.0
        for m in range(kh2):
            for n in range(kw2):
                sum_x += kernel_px[kh2 - 1 - m, kw2 - 1 - n] * img_b0[i + m, j + n]
                sum_y += kernel_py[kh2 - 1 - m, kw2 - 1 - n] * img_b0[i + m, j + n]
        gx0[i, j] = sum_x
        gy0[i, j] = sum_y

gx0_norm = np.round(cv2.normalize(gx0, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
gy0_norm = np.round(cv2.normalize(gy0, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# ----------- Anchor at (1,1) -----------
pad_top1, pad_left1 = 1, 1
pad_bottom1, pad_right1 = kh2 - 1 - pad_top1, kw2 - 1 - pad_left1
img_b1 = cv2.copyMakeBorder(img, pad_top1, pad_bottom1, pad_left1, pad_right1, cv2.BORDER_CONSTANT)
gx1 = np.zeros_like(img_b1, dtype=np.float32)
gy1 = np.zeros_like(img_b1, dtype=np.float32)

for i in range(h):
    for j in range(w):
        sum_x = 0.0
        sum_y = 0.0
        for m in range(kh2):
            for n in range(kw2):
                sum_x += kernel_px[kh2 - 1 - m, kw2 - 1 - n] * img_b1[i + m, j + n]
                sum_y += kernel_py[kh2 - 1 - m, kw2 - 1 - n] * img_b1[i + m, j + n]
        gx1[i + pad_top1, j + pad_left1] = sum_x
        gy1[i + pad_top1, j + pad_left1] = sum_y

gx1_norm = np.round(cv2.normalize(gx1, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
gy1_norm = np.round(cv2.normalize(gy1, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# Plot Prewitt separately
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Task 2: Prewitt Operator (Manual Loops)", fontsize=16)

# Anchor (0,0)
axs[0,0].imshow(img, cmap='gray'); axs[0,0].set_title("Original"); axs[0,0].axis('off')
axs[0,1].imshow(img_b0, cmap='gray'); axs[0,1].set_title("Padded"); axs[0,1].axis('off')
axs[0,2].imshow(gx0_norm, cmap='gray'); axs[0,2].set_title("Gx"); axs[0,2].axis('off')

# Anchor (1,1)
axs[1,0].imshow(img, cmap='gray'); axs[1,0].set_title("Original"); axs[1,0].axis('off')
axs[1,1].imshow(img_b1, cmap='gray'); axs[1,1].set_title("Padded"); axs[1,1].axis('off')
axs[1,2].imshow(gx1_norm, cmap='gray'); axs[1,2].set_title("Gx"); axs[1,2].axis('off')

plt.tight_layout()
plt.show()
