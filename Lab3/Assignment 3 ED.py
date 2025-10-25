import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Erlang function
def erlang(k, meu):
    E = np.zeros(256, dtype=np.float64)
    denom = (meu ** k) * math.factorial(k - 1)
    for x in range(256):
        num = (x ** (k - 1)) * np.exp(-x / meu)
        E[x] = num / denom
    return E

# Parameters
IMG_PATH = "input_image.png"
k = 80
lambda_val = 0.6
mu = 1.0 / lambda_val
L = 256

# Read input image
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"{IMG_PATH} not found.")

height, width = img.shape
x = np.arange(L)

# Input histogram, PDF, CDF
hist_in = cv2.calcHist([img], [0], None, [L], [0,256]).flatten()
pdf_in = hist_in / (height * width)
cdf_in = np.cumsum(pdf_in)

# Target Erlang histogram
target_pdf = erlang(k, mu)
target_pdf /= target_pdf.sum()
target_cdf = np.cumsum(target_pdf)

# Mapping input -> target
mapping = np.zeros(L, dtype=np.uint8)
for r in range(L):
    idx = np.searchsorted(target_cdf, cdf_in[r], side='left')
    if idx >= L: idx = L-1
    mapping[r] = np.uint8(idx)

matched_img = mapping[img]

# Output histogram, PDF, CDF
hist_out = cv2.calcHist([matched_img], [0], None, [L], [0,256]).flatten()
pdf_out = hist_out / (height * width)
cdf_out = np.cumsum(pdf_out)

# --- Display input and output images separately ---
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(matched_img, cmap='gray', vmin=0, vmax=255)
plt.title("Output Image (Erlang Matching)")
plt.axis('off')
plt.tight_layout()
plt.show()

# --- 2x2 grid: Histograms and PDFs/CDFs ---
fig, axs = plt.subplots(2, 2, figsize=(14,10))

# Top-left: Input vs Output Histogram
axs[0,0].plot(x, hist_in, lw=1.5, color='blue', label='Input Histogram')
axs[0,0].plot(x, hist_out, lw=1.5, color='orange', label='Output Histogram')
axs[0,0].set_title("Input vs Output Histogram (Counts)")
axs[0,0].set_xlim([0,255])
axs[0,0].set_xlabel("Intensity")
axs[0,0].set_ylabel("Count")
axs[0,0].legend()
axs[0,0].grid(alpha=0.3)

# Top-right: Target Histogram separately
axs[0,1].plot(x, target_pdf*height*width, color='red', lw=1.5)
axs[0,1].fill_between(x, target_pdf*height*width, 0, color='red', alpha=0.3)
axs[0,1].set_title("Target Histogram (Erlang)")
axs[0,1].set_xlim([0,255])
axs[0,1].set_xlabel("Intensity")
axs[0,1].set_ylabel("Count")
axs[0,1].grid(alpha=0.3)

# Bottom-left: PDFs
axs[1,0].plot(x, pdf_in, lw=1.5, color='blue', label='Input PDF')
axs[1,0].plot(x, pdf_out, lw=1.5, color='orange', label='Output PDF')
axs[1,0].plot(x, target_pdf, lw=1.5, color='red', linestyle='--', label='Target PDF')
axs[1,0].set_title("PDFs")
axs[1,0].set_xlim([0,255])
axs[1,0].set_xlabel("Intensity")
axs[1,0].set_ylabel("Probability")
axs[1,0].legend()
axs[1,0].grid(alpha=0.3)

# Bottom-right: CDFs
axs[1,1].plot(x, cdf_in, lw=1.5, color='blue', label='Input CDF')
axs[1,1].plot(x, cdf_out, lw=1.5, color='orange', label='Output CDF')
axs[1,1].plot(x, target_cdf, lw=1.5, color='red', linestyle='--', label='Target CDF')
axs[1,1].set_title("CDFs")
axs[1,1].set_xlim([0,255])
axs[1,1].set_xlabel("Intensity")
axs[1,1].set_ylabel("Cumulative Probability")
axs[1,1].legend()
axs[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Done.")
