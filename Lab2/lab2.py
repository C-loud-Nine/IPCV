import numpy as np
import cv2
import matplotlib.pyplot as plt

def log_function(x, y, sigma):
    r_squared = x**2 + y**2
    return -(1 / (np.pi * sigma**4)) * (1 - r_squared / (2 * sigma**2)) * np.exp(-r_squared / (2 * sigma**2))

def generate_log_kernel(sigma):
    size = int(np.ceil(9 * sigma))
    if size % 2 == 0:
        size += 1

    center = size // 2
    kernel = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = log_function(x, y, sigma)
    
    return kernel

def detect_zero_crossing(log_image, threshold=10):
    h, w = log_image.shape
    edge_map = np.zeros((h, w), dtype=np.uint8)
    strength_map = np.zeros((h, w), dtype=np.float32)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            current_pixel = log_image[i, j]
            
            top = log_image[i-1, j]
            bottom = log_image[i+1, j]
            left = log_image[i, j-1]
            right = log_image[i, j+1]
            
            zero_cross_detected = False
            if (current_pixel * top < 0) or (current_pixel * bottom < 0) or \
               (current_pixel * left < 0) or (current_pixel * right < 0):
                zero_cross_detected = True
            
            if zero_cross_detected:
                zs = (abs(current_pixel - top) + 
                      abs(current_pixel - bottom) + 
                      abs(current_pixel - left) + 
                      abs(current_pixel - right))
                strength_map[i, j] = zs
                
                if zs > threshold:
                    edge_map[i, j] = 255
    
    return edge_map, strength_map

def apply_log_convolution(image, sigma):
    log_kernel = generate_log_kernel(sigma)
    log_image = cv2.filter2D(image.astype(np.float32), -1, log_kernel)
    return log_image, log_kernel

def main(image_path, sigma=1.0, threshold=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}!")

    log_image, log_kernel = apply_log_convolution(image, sigma)
    edge_map, strength_map = detect_zero_crossing(log_image, threshold)
    
    strength_map_viz = cv2.normalize(strength_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    
    axs[0, 1].imshow(log_kernel, cmap='gray')
    axs[0, 1].set_title('LoG Kernel')
    
    axs[1, 0].imshow(strength_map_viz, cmap='gray')
    axs[1, 0].set_title('Zero Cross Strength')
    
    axs[1, 1].imshow(edge_map, cmap='gray')
    axs[1, 1].set_title('Edge Map')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = 'Lena.jpg'
    main(image_path, sigma=1.5, threshold=8)