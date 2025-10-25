#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 00:53:35 2025

@author: suma
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def generate_LoG_kernel(sigma):
    size = int(9 * sigma) | 1  
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(-k, k+1):
        for j in range(-k, k+1):
            r2 = i**2 + j**2
            kernel[k+i, k+j] = (-1 / (math.pi * sigma**4)) * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    
    return kernel

# ------------------------------
# Step 2: Convolution
# ------------------------------
def apply_log_filter(img, log_kernel):
    img = img.astype(np.float32)
    conv_img = cv2.filter2D(img, -1, log_kernel)
    return conv_img

# ------------------------------
# Step 3: Zero-cross detection
# ------------------------------
def zero_crossing_4neighborhood(log_img):
    M, N = log_img.shape
    zero_cross_img = np.zeros_like(log_img, dtype=np.float32)
    zero_strength_img = np.zeros_like(log_img, dtype=np.float32)
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            neighbors = [log_img[i-1,j], log_img[i+1,j], log_img[i,j-1], log_img[i,j+1]]
            zc = any(log_img[i,j] * n < 0 for n in neighbors)
            if zc:
                zero_cross_img[i,j] = log_img[i,j]
                zero_strength_img[i,j] = sum(abs(log_img[i,j] - n) for n in neighbors)
            else:
                zero_cross_img[i,j] = 0
    return zero_cross_img, zero_strength_img

# ------------------------------
# Step 4: Thresholding
# ------------------------------
def thresholding(img, th):
    threshold_img = np.where(img > th, 255, 0).astype(np.uint8)
    return threshold_img

# ------------------------------
# Step 5: Robust Laplacian Edge Detection
# ------------------------------
def local_variance_cv(img, ksize=3):
    img = img.astype(np.float32)
    mean = cv2.blur(img, (ksize, ksize))
    mean_sq = cv2.blur(img**2, (ksize, ksize))
    var = mean_sq - mean**2
    pad = ksize // 2
    var[:pad,:] = 0
    var[-pad:,:] = 0
    var[:,:pad] = 0
    var[:,-pad:] = 0
    return var

def robust_laplacian_edge_detector(img, log_img, th_var):
    M, N = img.shape
    edges = np.zeros_like(img, dtype=np.uint8)
    var_img = local_variance_cv(img, 3)
    zero_cross_img, zero_strength_img = zero_crossing_4neighborhood(log_img)
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if zero_cross_img[i,j] != 0 and var_img[i,j] > th_var:
                edges[i,j] = 255
            else:
                edges[i,j] = 0
                
    return zero_cross_img, zero_strength_img, edges

# ------------------------------
# Main Execution
# ------------------------------
sigma = 1
log_kernel = generate_LoG_kernel(sigma)
img = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)

conv_img = apply_log_filter(img, log_kernel)
zero_cross_img, zero_strength_img = zero_crossing_4neighborhood(conv_img)
threshold_img = thresholding(zero_strength_img, th=10)
_, _, robust_edges = robust_laplacian_edge_detector(img, conv_img, th_var=150)

# Normalize images for display
norm_conv_img = cv2.normalize(conv_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
abs_zero_cross = np.abs(zero_cross_img)
norm_zero_cross_img = cv2.normalize(abs_zero_cross, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
norm_zero_strength_img = cv2.normalize(zero_strength_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display results including LoG filtered image
plt.figure(figsize=(25, 5)) 

images = [img, norm_conv_img, norm_zero_cross_img, norm_zero_strength_img, robust_edges]
titles = ["Original Image", "LoG Filtered Image", "Zero-cross Pixels", "Zero-cross Strength", "Laplacian Edges"]

for i, (im, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 5, i+1)  
    plt.imshow(im, cmap="gray")
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.axis("image")

plt.subplots_adjust(wspace=0.05)  
plt.show()

# Also show the LoG kernel
plt.figure(figsize=(4,3))
plt.imshow(log_kernel, cmap="gray")
plt.title("LoG Kernel")
plt.axis("off")
plt.show()