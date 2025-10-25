# A_grayscale_equalization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_pdf_cdf(channel, L=256, rng=(0,256)):
    hist = cv2.calcHist([channel], [0], None, [L], rng).flatten()
    total = float(channel.size)
    pdf = hist / total
    cdf = np.cumsum(pdf)
    return hist, pdf, cdf

# --- load grayscale image ---
img_path = "input_image.png"   # replace with your path
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"{img_path} not found")

# --- original hist/pdf/cdf ---
hist_orig, pdf_orig, cdf_orig = hist_pdf_cdf(img, L=256, rng=(0,256))
bins = np.arange(256)

# --- manual equalization (LUT from CDF) ---
# Compute transform function
cdf = cdf_orig.copy()
lut = np.floor(255 * cdf + 0.5).astype(np.uint8)   # mapping 0..255 -> 0..255
equalized_manual = cv2.LUT(img, lut)

# --- OpenCV equalizeHist for comparison ---
equalized_cv = cv2.equalizeHist(img)

# --- compute hist/pdf/cdf after manual equalization ---
hist_eq, pdf_eq, cdf_eq = hist_pdf_cdf(equalized_manual, L=256, rng=(0,256))
hist_cv, pdf_cv, cdf_cv = hist_pdf_cdf(equalized_cv, L=256, rng=(0,256))

# --- plotting ---
plt.figure(figsize=(12,16))

# Row 1: original and manual equalized (images)
plt.subplot(3,2,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Original (grayscale)"); plt.axis('off')

plt.subplot(3,2,2)
plt.imshow(equalized_manual, cmap='gray', vmin=0, vmax=255)
plt.title("Equalized (manual LUT)"); plt.axis('off')

# Row 2: histograms (counts) original vs manual
plt.subplot(3,2,3)
plt.plot(bins, hist_orig, lw=1)
plt.title("Histogram (counts) - Original"); plt.xlim([0,255])
plt.xlabel('Intensity'); plt.ylabel('Count')

plt.subplot(3,2,4)
plt.plot(bins, hist_eq, lw=1)
plt.title("Histogram (counts) - Equalized (manual)"); plt.xlim([0,255])
plt.xlabel('Intensity'); plt.ylabel('Count')

# Row 3: PDF and CDF comparison (original vs manual)
plt.subplot(3,2,5)
plt.plot(bins, pdf_orig, label='Input PDF', lw=1)
plt.plot(bins, pdf_eq, label='Equalized PDF', lw=1)
plt.title("PDF - Input vs Equalized (manual)")
plt.xlabel('Intensity'); plt.ylabel('Probability'); plt.legend()

plt.subplot(3,2,6)
plt.plot(bins, cdf_orig, label='Input CDF', lw=1)
plt.plot(bins, cdf_eq, label='Equalized CDF', lw=1)
plt.title("CDF - Input vs Equalized (manual)")
plt.xlabel('Intensity'); plt.ylabel('Cumulative Probability'); plt.legend()

plt.tight_layout()
plt.show()
