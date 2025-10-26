import cv2
import numpy as np
from math import pi, sqrt
import os

# -----------------------------
# Shape descriptor functions
# -----------------------------
def compute_area(binary_img):
    return np.count_nonzero(binary_img)

def compute_border(binary_img):
    se = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_img, se, iterations=1)
    border = binary_img - eroded
    return border

def compute_perimeter(border_img):
    return np.count_nonzero(border_img)

def compute_max_diameter(binary_img):
    coords = np.argwhere(binary_img > 0)
    if coords.size == 0:
        return 1  # avoid division by zero
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    return max(max_x - min_x, max_y - min_y)

def compute_descriptors(binary_img):
    border = compute_border(binary_img)
    area = compute_area(binary_img)
    perimeter = compute_perimeter(border)
    max_d = compute_max_diameter(binary_img)
    form_factor = (4 * pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    roundness = (4 * area) / (pi * max_d ** 2) if max_d != 0 else 0
    compactness = (perimeter ** 2) / area if area != 0 else 0
    return form_factor, roundness, compactness

def euclidean_distance(desc1, desc2):
    return sqrt((desc1[0]-desc2[0])**2 + (desc1[1]-desc2[1])**2 + (desc1[2]-desc2[2])**2)

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
        images.append(img)
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
train_desc = [compute_descriptors(img) for img in train_images]
test_desc  = [compute_descriptors(img) for img in test_images]

# -----------------------------
# Print similarity table
# -----------------------------
row_headers = [f'Test {i+1}' for i in range(len(test_images))]
col_headers = [f'GT {i+1}' for i in range(len(train_images))]

print("\t" + "\t".join(col_headers))
for i, td in enumerate(test_desc):
    distances = [euclidean_distance(td, tr) for tr in train_desc]
    print(f"{row_headers[i]}\t" + "\t".join(f"{d:.2f}" for d in distances))
