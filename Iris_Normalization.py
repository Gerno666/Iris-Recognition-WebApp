import cv2
import numpy as np
import matplotlib.pyplot as plt
from Iris_Localization import IrisLocalization
from Eyelid_Detection import EyelidDetection, draw_bezier_curve

def rubber_sheet_normalization(image, iris_center, iris_radius, pupil_radius, upper_eyelid_curve=None, lower_eyelid_curve=None, output_height=64, output_width=512):
    iris_center_x, iris_center_y = iris_center
    
    # Definizione delle coordinate polari
    theta = np.linspace(0, 2 * np.pi, output_width)
    r = np.linspace(pupil_radius, iris_radius, output_height)
    theta, r = np.meshgrid(theta, r)

    # Conversione delle coordinate polari in coordinate cartesiane
    x = iris_center_x + r * np.cos(theta)
    y = iris_center_y + r * np.sin(theta)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Creazione della maschera iniziale (tutti i pixel validi)
    mask = np.ones_like(x, dtype=np.uint8)

    # Aggiornamento della maschera per escludere le regioni delle palpebre
    if upper_eyelid_curve is not None:
        for i in range(output_width):
            for j in range(output_height):
                if y[j, i] < np.interp(x[j, i], upper_eyelid_curve[0], upper_eyelid_curve[1]):
                    mask[j, i] = 0

    if lower_eyelid_curve is not None:
        for i in range(output_width):
            for j in range(output_height):
                if y[j, i] > np.interp(x[j, i], lower_eyelid_curve[0], lower_eyelid_curve[1]):
                    mask[j, i] = 0

    # Applicazione della mappatura delle coordinate
    normalized_iris = cv2.remap(image, x, y, cv2.INTER_LINEAR)
    
    # Calcolo del valore medio dei pixel validi
    mean_value = cv2.mean(normalized_iris, mask=mask)[0]
    
    # Riempimento delle aree mascherate con il valore medio
    normalized_iris = cv2.bitwise_and(normalized_iris, normalized_iris, mask=mask)
    normalized_iris[mask == 0] = mean_value

    return normalized_iris

def normalize_iris(images, boundaries, upper_eyelid_curves, lower_eyelid_curves):

    normalized_iris_imgs = []

    # Disegna le curve di Bézier e mostra i risultati
    for image, boundary, upper_curve, lower_curve in zip(images, boundaries, upper_eyelid_curves, lower_eyelid_curves):
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

        # Normalizzazione dell'iride
        normalized_iris = rubber_sheet_normalization(image, (x, y), radius_pupil + 55, radius_pupil, upper_eyelid_curve=upper_curve, lower_eyelid_curve=lower_curve, output_height=64, output_width=512)

        # Riempire i bordi bianchi con il colore medio dell'iride
        mean_value = np.mean(normalized_iris[normalized_iris > 0])
        normalized_iris[normalized_iris == 255] = mean_value

        normalized_iris_imgs.append(normalized_iris)

    return normalized_iris_imgs



'''# Caricamento dell'immagine
image_path = 'data/CASIA Iris Image Database (version 1.0)/002/1/002_1_1.bmp'
image = cv2.imread(image_path)

# Esecuzione di IrisLocalization
img, boundaries, filtered_images = IrisLocalization([image])

# Immagine con palpebra localizzata
img, boundaries, upper_eyelid_curves, lower_eyelid_curves = EyelidDetection([img[0]], boundaries)

# Normalizzazione dell'iride
normalized_iris_imgs = normalize_iris([image], boundaries, upper_eyelid_curves, lower_eyelid_curves)
normalized_iris = normalized_iris_imgs[0]

# Visualizzazione dell'iride normalizzata
plt.figure(figsize=(15, 5))

plt.subplot(1,2,2)
plt.title('Normalized Iris')
plt.imshow(cv2.cvtColor(normalized_iris, cv2.COLOR_BGR2RGB))
plt.axis('off')

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
plt.subplot(1, 2, 1)
plt.title('Localized Eyelid')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()'''