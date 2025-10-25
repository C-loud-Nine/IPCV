import cv2
import numpy as np
from math import sqrt, pi
import os

# -----------------------------
# Object geometry and descriptor functions
# -----------------------------
def obj_geometry(bin_img):
    area = np.count_nonzero(bin_img)
    se = np.ones((3,3), np.uint8)
    eroded = cv2.erode(bin_img, se, iterations=1)
    border = bin_img - eroded
    perimeter = np.count_nonzero(border)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    a, b = 1, 1
    if len(cnt) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        a = max(MA, ma)
        b = min(MA, ma)
    return area, perimeter, (a, b)

def descriptors(binary_img):
    area, perimeter, (a, b) = obj_geometry(binary_img)
    form_factor = (4 * pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    compactness = (perimeter ** 2) / area if area != 0 else 0
    eccentricity = sqrt(1 - (b/a)**2) if a != 0 else 0
    return np.array([form_factor, compactness, eccentricity], dtype=np.float64)

# -----------------------------
# Distance functions
# -----------------------------
def kl_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    eps = 1e-10
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def euclidean_distance(p, q):
    return sqrt(np.sum((p - q)**2))

def cosine_similarity(p, q):
    return np.sum(p * q) / (sqrt(np.sum(p**2)) * sqrt(np.sum(q**2)))

def calculate_distances(test_desc, train_desc, distance_func):
    return [distance_func(test_desc, td) for td in train_desc]

# -----------------------------
# Load images safely
# -----------------------------
def load_images(paths):
    images = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv2.imread(p, 0)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        images.append(binary)
    return images

train_paths = [
    "assets/c1.jpg",
    "assets/t1.jpg",
    "assets/p1.png"
]

test_paths = [
    "assets/c2.jpg",
    "assets/t2.jpg",
    "assets/p2.png",
    "assets/st.jpg"
]

train_images = load_images(train_paths)
test_images = load_images(test_paths)

# -----------------------------
# Compute descriptors
# -----------------------------
train_desc = [descriptors(img) for img in train_images]
test_desc  = [descriptors(img) for img in test_images]

# -----------------------------
# Print KL Divergence table
# -----------------------------
print("KL Divergence Table:")
row_headers = [f'Test {i+1}' for i in range(len(test_images))]
col_headers = [f'GT {i+1}' for i in range(len(train_images))]
print("\t" + "\t".join(col_headers))
for i, td in enumerate(test_desc):
    distances = calculate_distances(td, train_desc, kl_divergence)
    print(f"{row_headers[i]}\t" + "\t".join(f"{d:.4f}" for d in distances))

# -----------------------------
# Print Euclidean Distance table
# -----------------------------
print("\nEuclidean Distance Table:")
print("\t" + "\t".join(col_headers))
for i, td in enumerate(test_desc):
    distances = calculate_distances(td, train_desc, euclidean_distance)
    print(f"{row_headers[i]}\t" + "\t".join(f"{d:.4f}" for d in distances))

# -----------------------------
# Print Cosine Similarity table
# -----------------------------
print("\nCosine Similarity Table:")
print("\t" + "\t".join(col_headers))
for i, td in enumerate(test_desc):
    distances = calculate_distances(td, train_desc, cosine_similarity)
    print(f"{row_headers[i]}\t" + "\t".join(f"{d:.4f}" for d in distances))
