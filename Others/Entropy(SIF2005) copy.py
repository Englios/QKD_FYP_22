import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color,exposure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border, mark_boundaries
from scipy.stats import entropy 

# Load the image
img = io.imread("C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/20.jpg")

# Convert the image to grayscale
gray_img = color.rgb2gray(exposure.adjust_gamma(img,1,2))

# Threshold the image using Otsu's method
threshold = threshold_otsu(gray_img)
binary_img = gray_img > threshold
binary_img=~binary_img

# Remove border objects
cleared_img = clear_border(binary_img)

# Create a mask of the green regions
green_mask = np.logical_not(img[..., 1] > 100, img[..., 2] < 150)

# Apply the mask to the cleared image
masked_img = np.zeros_like(cleared_img)
masked_img[green_mask] = cleared_img[green_mask]

# Plot the original image, binary image, and masked image
fig, ax = plt.subplots(ncols=4, figsize=(8, 4))

ax[0].imshow(img)
ax[0].set_title("Original Image")

ax[3].imshow(gray_img,cmap="gray")
ax[3].set_title("Gray Image")

ax[1].imshow(binary_img, cmap="gray")
ax[1].set_title("Binary Image")

ax[2].imshow(mark_boundaries(img, masked_img))
ax[2].set_title("Masked Image")

plt.show()
