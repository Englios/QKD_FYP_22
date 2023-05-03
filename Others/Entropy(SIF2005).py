import numpy as np
from skimage import io, color,exposure
from scipy.stats import entropy
import matplotlib.pyplot as mpl
from scipy.ndimage import binary_fill_holes
import cv2


def green_entropy(img):
    og=img.copy() #Original Unedited image

    # Adjust the brightness and contrast of the image
    alpha = 1.7  # Contrast control (1.0-3.0)
    beta = 50  # Brightness control (0-100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Create a mask of the green regions
    green_mask = np.logical_and.reduce((img[..., 1] > 40, img[..., 0] < 255, img[..., 2] < 255, 
                                        np.sum(img, axis=-1) > 20))
    mask=np.logical_not(green_mask)
    filled_mask=binary_fill_holes(mask)

    # Create a new 4-channel mask with transparency
    alpha_mask = np.zeros_like(filled_mask, dtype=np.float32)
    alpha_mask[filled_mask] = 0.5  # Set the alpha channel for the filled mask to 0.5

    # Apply the mask to the original image
    green_img = img.copy()
    green_img = (green_img * alpha_mask[..., np.newaxis] + 255 * (1 - alpha_mask[..., np.newaxis])).astype(np.uint8)
    og = (og * alpha_mask[..., np.newaxis] + 255 * (1 - alpha_mask[..., np.newaxis])).astype(np.uint8)

    # Convert the masked image to grayscale
    gray_img = color.rgb2gray(exposure.adjust_gamma(green_img,1,2))
    
    # Compute the probability distribution function (PDF) of the green pixels
    pdf, _ = np.histogram(gray_img[green_mask], bins=np.arange(0, 1.01, 0.01))
    pdf = pdf / pdf.sum()

    # Compute the entropy of the PDF
    green_entropy = entropy(pdf, base=2)
    
    return og, green_entropy

# Load the images
years=['2001',
       '2004',
       '2008',
       '2009',
       '2010',
       '2010(2)',
       '2012',
       '2014',
       '2015',
       '2015(2)',
       '2016',
       '2017',
       '2017(2)',
       '2018',
       '2019',
       '2020',
       '2020(2)',
       '2021']
image_paths = ["C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2001.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2004.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2008.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2009.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2010_2.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2010.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2012.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2014.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2015_2.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2015.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2016.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2017_2.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2017.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2018.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2019.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2020_2.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2020.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2021.jpg"]

images = [io.imread(path) for path in image_paths]

entropies=[]
masked_images=[]
# Compute the entropy of the green pixels in each image
for img in images:
    masked_image, entropy_val = green_entropy(img)
    masked_images.append(masked_image)
    entropies.append(entropy_val)

print("Entropy values for each image:", entropies)
mpl.scatter(years,entropies)
mpl.show()

#PLOTTING
fig = mpl.figure(figsize=(12, 16))
columns = 4
rows = 5

ax = []
for i in range(columns*rows):
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title(years[i-2]) 
    ax[-1].axis('off')
    ax[-1].set_aspect("equal")
    mpl.imshow(masked_images[i-2])
    
mpl.show()
