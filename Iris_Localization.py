import cv2
import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt

def estimate_center(img):
    horizontal_projection = np.mean(img, 0)
    vertical_projection = np.mean(img, 1)
    center_x = horizontal_projection.argmin()
    center_y = vertical_projection.argmin()
    return center_x, center_y

def IrisLocalization(images):
    boundaries = []  # List to hold the centers of the boundary circles
    filtered_images = []  # List to hold filtered images

    for img in images:
        # Convert image to grayscale if it is not already
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove noise by blurring the image
        filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
        filtered_images.append(filtered_img)
        
        # Estimate the center of the pupil
        center_x, center_y = estimate_center(filtered_img)
        
        # Recalculate the center of the pupil by concentrating on a 120x120 area
        half_crop_size = 60
        if center_x > half_crop_size and center_y > half_crop_size and center_x + half_crop_size < filtered_img.shape[1] and center_y + half_crop_size < filtered_img.shape[0]:
            cropped_img = filtered_img[center_y - half_crop_size:center_y + half_crop_size, center_x - half_crop_size:center_x + half_crop_size]
            crop_center_x, crop_center_y = estimate_center(cropped_img)
            crop_center_x += center_x - half_crop_size
            crop_center_y += center_y - half_crop_size
        else:
            crop_center_x, crop_center_y = center_x, center_y

        # Apply Canny edge detector on the masked image
        mask_image = cv2.inRange(filtered_img, 0, 70)
        output = cv2.bitwise_and(filtered_img, filtered_img, mask=mask_image)
        edged = cv2.Canny(output, 100, 220)
        
        # Apply Hough transform to find potential boundaries of pupil
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 10, 100, param1=50, param2=30, minRadius=0, maxRadius=0)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            min_dst = math.inf
            best_circle = None

            for i in circles[0, :]:
                # Find the circle whose center is closest to the approx center found above
                dst = distance.euclidean((crop_center_x, crop_center_y), (i[0], i[1]))
                if dst < min_dst:
                    min_dst = dst
                    best_circle = i

            if best_circle is not None:
                x, y, radius_pupil = best_circle
                radius_pupil = int(radius_pupil)
                
                boundaries.append([x, y, radius_pupil])

    return images, boundaries, filtered_images


'''# Caricamento dell'immagine
image_path = 'data/CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp'
image = cv2.imread(image_path)

# Esecuzione di IrisLocalization
img, boundaries, filtered_images = IrisLocalization([image])

# Visualizzazione del risultato
plt.figure(figsize=(15, 5))

# Immagine originale
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Immagine con filtro bilaterale
plt.subplot(1, 3, 2)
plt.title('Filtered Image')
plt.imshow(filtered_images[0], cmap='gray')
plt.axis('off')

# Immagine con iride localizzata
if boundaries:
    img = img[0]
    x, y, radius_pupil = boundaries[0]
    # Draw the inner boundary
    cv2.circle(img, (x, y), radius_pupil, (255, 255, 255), 2)
    # Draw the outer boundary, approximately 55 pixels from the inner boundary
    cv2.circle(img, (x, y), radius_pupil + 55, (255, 255, 255), 2)
    plt.subplot(1, 3, 3)
    plt.title('Localized Iris')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
else:
    print("No circles were found!")

plt.show()'''