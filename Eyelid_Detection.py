import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
from Iris_Localization import IrisLocalization

def detect_eyelid(cropped_img, x_offset, y_offset, radius_pupil, x, y, above=True):
    edges = cv2.Canny(cropped_img, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
    
    points = []
    best_line = None
    min_dst = math.inf

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Adjust coordinates to the original image
                x1 += x_offset
                y1 += y_offset
                x2 += x_offset
                y2 += y_offset

                midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
                dst = distance.euclidean((x, y), midpoint)

                if dst >= radius_pupil + 10:
                    if (above and y1 < y and y2 < y) or (not above and y1 > y and y2 > y and min(y1, y2) > y + radius_pupil):
                        if dst < min_dst:
                            min_dst = dst
                            best_line = (x1, y1, x2, y2)
                            points.append((x1, y1))
                            points.append((x2, y2))
    return best_line, points

def calculate_bezier_curve(x1, y1, x2, y2, curve_height):
    # Punto medio della linea
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2

    # Calcolo dell'angolo per il punto di controllo
    angle = np.pi / 2
    
    # Calcolo del punto di controllo
    cx = mx + curve_height * np.cos(angle)
    cy = my + curve_height * np.sin(angle)

    # Creazione dei punti della curva usando una curva di Bézier quadratica
    t = np.linspace(0, 1, 100)
    xt = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
    yt = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2

    return xt, yt

def draw_bezier_curve(image, xt, yt, color, thickness):
    for i in range(len(xt) - 1):
        cv2.line(image, (int(xt[i]), int(yt[i])), (int(xt[i+1]), int(yt[i+1])), color, thickness)

def EyelidDetection(images, boundaries):
    eyelid_boundaries = []
    upper_eyelid_curves = []
    lower_eyelid_curves = []

    for img, boundary in zip(images, boundaries):
        x, y, radius_pupil = boundary

        # Convert image to grayscale if it is not already
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Crop the region above the eye
        half_crop_size = int(radius_pupil * 2)
        crop_x1 = max(0, x - 55)
        crop_x2 = min(img.shape[1], x + 55)

        # Region above the pupil
        crop_y1_top = max(0, y - half_crop_size)
        crop_y2_top = y - (radius_pupil + 5)
        cropped_img_top = img[crop_y1_top:crop_y2_top, crop_x1:crop_x2]

        # Region below the pupil
        crop_y1_bottom = y + (radius_pupil - 5)
        crop_y2_bottom = min(img.shape[0], y + half_crop_size)
        cropped_img_bottom = img[crop_y1_bottom:crop_y2_bottom, crop_x1:crop_x2]

        best_line_top, points_top = detect_eyelid(cropped_img_top, crop_x1, crop_y1_top, radius_pupil, x, y, above=True)
        best_line_bottom, points_bottom = detect_eyelid(cropped_img_bottom, crop_x1, crop_y1_bottom, radius_pupil, x, y, above=False)

        if best_line_top is not None or best_line_bottom is not None:
            eyelid_boundaries.append((best_line_top, best_line_bottom))

            if best_line_top is not None:
                x1, y1, x2, y2 = best_line_top
                a = x1 + (x2 - x1) / 2
                x1 = a - 200
                x2 = a + 200
                if y1 > y2:
                    y1 += 50
                    y2 = y1
                else:
                    y2 += 50
                    y1 = y2
                xt, yt = calculate_bezier_curve(int(x1), y1, int(x2), y2, curve_height=-100)
                upper_eyelid_curves.append((xt, yt))
            else:
                upper_eyelid_curves.append(None)
            
            if best_line_bottom is not None:
                x1, y1, x2, y2 = best_line_bottom
                a = x1 + (x2 - x1) / 2
                x1 = a - 200
                x2 = a + 200
                if y1 > y2:
                    y2 -= 50
                    y1 = y2
                else:
                    y1 -= 50
                    y2 = y1
                xt, yt = calculate_bezier_curve(int(x1), y1, int(x2), y2, curve_height=100)
                lower_eyelid_curves.append((xt, yt))
            else:
                lower_eyelid_curves.append(None)
        else:
            upper_eyelid_curves.append(None)
            lower_eyelid_curves.append(None)

    return images, boundaries, upper_eyelid_curves, lower_eyelid_curves


'''# Caricamento dell'immagine
image_path = 'data/CASIA Iris Image Database (version 1.0)/002/1/002_1_1.bmp'
image = cv2.imread(image_path)

# Esecuzione di IrisLocalization
img, boundaries, filtered_images = IrisLocalization([image])

# Visualizzazione del risultato
plt.figure(figsize=(10, 5))

# Immagine con iride localizzata
if boundaries:
    img = img[0]
    x, y, radius_pupil = boundaries[0]
    # Draw the inner boundary
    cv2.circle(img, (x, y), radius_pupil, (255, 255, 255), 2)
    # Draw the outer boundary, approximately 55 pixels from the inner boundary
    cv2.circle(img, (x, y), radius_pupil + 55, (255, 255, 255), 2)
    plt.subplot(1, 2, 1)
    plt.title('Localized Iris')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
else:
    print("No circles were found!")


# Caricamento dell'immagine
image_path = 'data/CASIA Iris Image Database (version 1.0)/002/1/002_1_1.bmp'
image = cv2.imread(image_path)

# Immagine con palpebra localizzata
img, boundaries, upper_eyelid_curves, lower_eyelid_curves = EyelidDetection([image], boundaries)

# Disegna le curve di Bézier e mostra i risultati
for boundary, upper_curve, lower_curve in zip(boundaries, upper_eyelid_curves, lower_eyelid_curves):
    if boundary is not None:
        x, y, radius_pupil = boundary
        # Disegna il cerchio della pupilla
        cv2.circle(image, (x, y), radius_pupil, (255, 255, 255), 2)
        cv2.circle(image, (x, y), radius_pupil + 55, (255, 255, 255), 2)

    if upper_curve is not None:
        xt, yt = upper_curve
        draw_bezier_curve(image, xt, yt, color=(255, 255, 255), thickness=2)
    
    if lower_curve is not None:
        xt, yt = lower_curve
        draw_bezier_curve(image, xt, yt, color=(255, 255, 255), thickness=2)

# Mostra l'immagine con le linee delle palpebre e il cerchio della pupilla
plt.subplot(1, 2, 2)
plt.title('Localized Eyelid')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()'''