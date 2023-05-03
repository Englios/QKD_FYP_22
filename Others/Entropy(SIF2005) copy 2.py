import numpy as np
from skimage import io, color,exposure
from scipy.stats import entropy
import matplotlib.pyplot as mpl
from scipy.ndimage import binary_fill_holes
import cv2


def green_entropy(img):
    # Adjust the brightness and contrast of the image
    alpha = 1.7  # Contrast control (1.0-3.0)
    beta = 50  # Brightness control (0-100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    og= cv2.convertScaleAbs(img, alpha=0.8, beta=0)
    
    
    # Create a mask of the green regions
    green_mask = np.logical_and.reduce((img[..., 1] > 40, img[..., 0] < 255, img[..., 2] < 255, 
                                        np.sum(img, axis=-1) > 20))
    mask=np.logical_not(green_mask)
    filled_mask=binary_fill_holes(mask)

    # Apply the mask to the original image
    green_img = img.copy()
    green_img[~filled_mask] = [53,254,162]
    
    og[filled_mask] = [0]
    

    # Convert the masked image to grayscale
    gray_img = color.rgb2gray(exposure.adjust_gamma(green_img,1,2))
    
    # Compute the probability distribution function (PDF) of the green pixels
    pdf, _ = np.histogram(gray_img[green_mask], bins=np.arange(0, 1.01, 0.01))
    pdf = pdf / pdf.sum()

    # Compute the entropy of the PDF
    green_entropy = entropy(pdf, base=2)
    
    # Create a boundary mask from the filled mask
    boundary_mask = cv2.Canny(filled_mask.astype(np.uint8)*255, 100, 200)
    
    # Draw the boundary on the original image
    contours, hierarchy = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(~og, contours, -1, (0, 255, 0), 1)

    
    
    return og, green_entropy

# Load the images
years_2=["'01",
       "'04",
       "'08",
       "'09",
       "'10",
       "'10",
       "'12",
       "'14",
       "'15",
       "'15",
       "'16",
       "'17",
       "'17",
       "'18",
       "'19",
       "'20",
       "'21"]
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
       '2021']
image_paths = ["C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2001.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2004.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2008.jpg",
               "C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Others/Green Patch Img/2009_2.jpeg",
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
mpl.bar(years_2,entropies)
mpl.xlabel("Years(20')")
mpl.ylabel("Shannon Entropy")
mpl.title("Entropy of Green Patches over 2 Decades (2001-2021)")
mpl.show()

# #PLOTTING
# fig = mpl.figure(figsize=(15, 16))
# fig.subplots_adjust(hspace=0.5, wspace=0.1)
# columns = 4
# rows = 5

# ax = []
# for i in range(columns*rows):
#     # create subplot and append to ax
#     ax.append( fig.add_subplot(rows, columns, i+1) )
#     title = f"{years[i-3]} - Entropy: {entropies[i-3]:.3f}"
#     ax[-1].set_title(title) 
#     ax[-1].axis('off')
#     ax[-1].set_aspect("equal")
#     mpl.imshow(masked_images[i-3])
    
# mpl.show()
fig = mpl.figure(figsize=(15, 16))
fig.subplots_adjust(hspace=0.5, wspace=0.1)
columns = 4
rows = 5

ax = []
for i in range(columns*rows):
    # skip the first three columns of the first row
    if i < 3:
        continue

    # calculate the index in the masked_images array
    img_index = i - 3 if i >= 4 else i - 3
    title = f"{years[img_index]} - Entropy: {entropies[img_index]:.3f}"
    
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title(title) 
    ax[-1].axis('off')
    ax[-1].set_aspect("equal")
    mpl.imshow(masked_images[img_index])
    
mpl.show()
