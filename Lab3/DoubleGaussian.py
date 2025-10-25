#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 00:50:14 2025

@author: suma
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Load a grayscale input image
img_gray = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)
x = np.arange(256)
x = np.arange(256)
# Define means and standard deviations
mean1, std1 = 90, 30
mean2, std2 = 180, 20

# Compute Gaussian distributions
gaussian1 = (1 / (std1 * np.sqrt(2 * np.pi))) * np.exp(-((x - mean1) ** 2) / (2 * (std1 ** 2)))
gaussian2 = (1 / (std2 * np.sqrt(2 * np.pi))) * np.exp(-((x - mean2) ** 2) / (2 * (std2 ** 2)))

target_pdf = gaussian1 + gaussian2
# Normalize target PDF so its sum is 1
target_pdf /= target_pdf.sum()

# Convert PDF to target histogram 
target_hist = target_pdf * img_gray.size
target_hist = target_hist.astype(np.uint32)

# Create the target image by mapping values based on the histogram
target_image = np.zeros_like(img_gray.flatten())
pointer = 0
for i in range(256):
    count = target_hist[i]
    target_image[pointer:pointer+count] = i
    pointer += count
target_image = target_image.reshape(img_gray.shape)

matched = match_histograms(img_gray, target_image, channel_axis=None)

# === Step 5: Plot input and matched images ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Input Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(matched, cmap='gray')
plt.title('Histogram Matched Image')
plt.axis('off')
plt.tight_layout()
plt.show()

def plot_hist_pdf_cdf(image, title_prefix):
    image_flat = image.flatten()
    
    hist, bins = np.histogram(image_flat, bins=256, range=[0, 256])
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)

    return hist, pdf, cdf

# Calculate stats
hist_input, pdf_input, cdf_input = plot_hist_pdf_cdf(img_gray, 'Input')
hist_matched, pdf_matched, cdf_matched = plot_hist_pdf_cdf(matched, 'Matched')

plt.figure(figsize=(18, 12))

# Compute target CDF and scaled histogram
target_cdf = np.cumsum(target_pdf)
target_cdf /= target_cdf[-1]
target_hist_scaled = target_pdf * np.prod(img_gray.shape)

# ------------------ Row 1: Histogram ------------------
plt.subplot(3, 3, 1)
plt.plot(hist_input, color='blue')
plt.title("Input Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(3, 3, 2)
plt.bar(x, target_hist_scaled, color='gray')
plt.title("Histogram After Applying Double Gaussian")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(3, 3, 3)
plt.plot(hist_matched, color='green')
plt.title("Matched Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# ------------------ Row 2: PDF ------------------
plt.subplot(3, 3, 4)
plt.plot(pdf_input, color='blue')
plt.title("Input PDF")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")

plt.subplot(3, 3, 5)
plt.plot(x, target_pdf, color='grey')
plt.title("PDF After Applying Double Gaussian")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")

plt.subplot(3, 3, 6)
plt.plot(pdf_matched, color='green')
plt.title("Matched PDF")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")

# ------------------ Row 3: CDF ------------------
plt.subplot(3, 3, 7)
plt.plot(cdf_input, color='blue')
plt.title("Input CDF")
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Probability")

plt.subplot(3, 3, 8)
plt.plot(target_cdf, color='grey')
plt.title("CDF After Applying Double Gaussian")
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Probability")

plt.subplot(3, 3, 9)
plt.plot(cdf_matched, color='green')
plt.title("Matched CDF")
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Probability")

plt.tight_layout(pad=3.0)
plt.show()

