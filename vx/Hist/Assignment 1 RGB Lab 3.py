# B_rgb_per_channel_equalization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_pdf_cdf(channel, L=256, rng=(0,256)):
    hist = cv2.calcHist([channel], [0], None, [L], rng).flatten()
    total = float(channel.size)
    pdf = hist / total
    cdf = np.cumsum(pdf)
    return hist, pdf, cdf

# --- load color image ---
img_path = "input_image.png"   # replace with your path
img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError(f"{img_path} not found")

# convert BGR -> RGB for nicer matplotlib display
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# split channels (note: these are RGB channels)
r_channel = img_rgb[:, :, 0]
g_channel = img_rgb[:, :, 1]
b_channel = img_rgb[:, :, 2]

# equalize each channel separately (OpenCV works on single-channel uint8)
r_eq = cv2.equalizeHist(r_channel)
g_eq = cv2.equalizeHist(g_channel)
b_eq = cv2.equalizeHist(b_channel)

# merge equalized channels back to RGB
rgb_eq = cv2.merge([r_eq, g_eq, b_eq])

# --- show channel splits before and after equalization ---
plt.figure(figsize=(12,8))
# Original channels
plt.subplot(2,3,1); plt.imshow(r_channel, cmap='gray'); plt.title("R Channel Original"); plt.axis('off')
plt.subplot(2,3,2); plt.imshow(g_channel, cmap='gray'); plt.title("G Channel Original"); plt.axis('off')
plt.subplot(2,3,3); plt.imshow(b_channel, cmap='gray'); plt.title("B Channel Original"); plt.axis('off')
# Equalized channels
plt.subplot(2,3,4); plt.imshow(r_eq, cmap='gray'); plt.title("R Channel Equalized"); plt.axis('off')
plt.subplot(2,3,5); plt.imshow(g_eq, cmap='gray'); plt.title("G Channel Equalized"); plt.axis('off')
plt.subplot(2,3,6); plt.imshow(b_eq, cmap='gray'); plt.title("B Channel Equalized"); plt.axis('off')
plt.tight_layout(); plt.show()

# --- plotting histograms and CDFs remains the same as your original code ---
bins = np.arange(256)
hist_r, pdf_r, cdf_r = hist_pdf_cdf(r_channel)
hist_g, pdf_g, cdf_g = hist_pdf_cdf(g_channel)
hist_b, pdf_b, cdf_b = hist_pdf_cdf(b_channel)
hist_r_eq, pdf_r_eq, cdf_r_eq = hist_pdf_cdf(r_eq)
hist_g_eq, pdf_g_eq, cdf_g_eq = hist_pdf_cdf(g_eq)
hist_b_eq, pdf_b_eq, cdf_b_eq = hist_pdf_cdf(b_eq)

plt.figure(figsize=(14,18))
plt.subplot(4,2,1); plt.imshow(img_rgb); plt.title("Original (RGB)"); plt.axis('off')
plt.subplot(4,2,2); plt.imshow(rgb_eq); plt.title("Equalized per-channel (RGB)"); plt.axis('off')

plt.subplot(4,2,3); plt.plot(bins, hist_r, color='red', label='Hist Before'); plt.plot(bins, hist_r_eq, color='orange', label='Hist After'); plt.title('Red Channel Histogram'); plt.legend(); plt.xlim([0,255])
plt.subplot(4,2,4); plt.plot(bins, cdf_r, color='red', label='CDF Before'); plt.plot(bins, cdf_r_eq, color='orange', label='CDF After'); plt.title('Red Channel CDF'); plt.legend(); plt.xlim([0,255])
plt.subplot(4,2,5); plt.plot(bins, hist_g, color='green', label='Hist Before'); plt.plot(bins, hist_g_eq, color='lime', label='Hist After'); plt.title('Green Channel Histogram'); plt.legend(); plt.xlim([0,255])
plt.subplot(4,2,6); plt.plot(bins, cdf_g, color='green', label='CDF Before'); plt.plot(bins, cdf_g_eq, color='lime', label='CDF After'); plt.title('Green Channel CDF'); plt.legend(); plt.xlim([0,255])
plt.subplot(4,2,7); plt.plot(bins, hist_b, color='blue', label='Hist Before'); plt.plot(bins, hist_b_eq, color='cyan', label='Hist After'); plt.title('Blue Channel Histogram'); plt.legend(); plt.xlim([0,255])
plt.subplot(4,2,8); plt.plot(bins, cdf_b, color='blue', label='CDF Before'); plt.plot(bins, cdf_b_eq, color='cyan', label='CDF After'); plt.title('Blue Channel CDF'); plt.legend(); plt.xlim([0,255])
plt.tight_layout(); plt.show()
