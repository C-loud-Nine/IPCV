# C_hsv_value_equalization.py
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

# convert to RGB for display
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# convert to HSV and split
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv2.split(hsv)

# equalize only the Value (V) channel
v_eq = cv2.equalizeHist(v_channel)

# merge back the equalized channel
hsv_eq = cv2.merge([h_channel, s_channel, v_eq])
rgb_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

# Compute hist/pdf/cdf for Value channel (before & after)
hist_v, pdf_v, cdf_v = hist_pdf_cdf(v_channel, L=256, rng=(0,256))
hist_v_eq, pdf_v_eq, cdf_v_eq = hist_pdf_cdf(v_eq, L=256, rng=(0,256))

# --- visualization ---
plt.figure(figsize=(12,12))

# Row 1: Original RGB and Enhanced RGB
plt.subplot(3,2,1)
plt.imshow(img_rgb)
plt.title("Original Image (RGB)")
plt.axis('off')

plt.subplot(3,2,2)
plt.imshow(rgb_eq)
plt.title("Enhanced Image (HSV - V Equalized)")
plt.axis('off')

# Row 2: Value channel before and after equalization
plt.subplot(3,2,3)
plt.imshow(v_channel, cmap='gray')
plt.title("Value Channel (Before)")
plt.axis('off')

plt.subplot(3,2,4)
plt.imshow(v_eq, cmap='gray')
plt.title("Value Channel (After Equalization)")
plt.axis('off')

# Row 3: Histograms and CDFs
bins = np.arange(256)

plt.subplot(3,2,5)
plt.plot(bins, hist_v, color='blue', label='Before')
plt.plot(bins, hist_v_eq, color='cyan', label='After')
plt.title('Value Channel Histogram (Counts)')
plt.legend()
plt.xlim([0,255])

plt.subplot(3,2,6)
plt.plot(bins, cdf_v, color='blue', label='Before')
plt.plot(bins, cdf_v_eq, color='cyan', label='After')
plt.title('Value Channel CDF')
plt.legend()
plt.xlim([0,255])

plt.tight_layout()
plt.show()
