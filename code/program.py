
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_path = "C:/Users/ashle/OneDrive/Desktop/AstraSage/synthetic_lunar_crater.png"


img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not loaded. Check the file path and file name.")
    exit()


norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("float32")

gy, gx = np.gradient(norm_img)
gradient_magnitude = np.sqrt(gx**2 + gy**2)


heightmap = 255 - norm_img  # Inverse brightness to estimate elevation

heightmap = cv2.normalize(heightmap, None, 0, 255, cv2.NORM_MINMAX)
heightmap = heightmap.astype(np.uint8)

output_path = os.path.join(os.path.dirname(image_path), "lunar_heightmap.png")
cv2.imwrite(output_path, heightmap)

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(norm_img, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Generated Heightmap")
plt.imshow(heightmap, cmap="gray")

plt.show()
