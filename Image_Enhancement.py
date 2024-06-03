import cv2
import numpy as np
import matplotlib.pyplot as plt
from Iris_Localization import IrisLocalization
from Eyelid_Detection import EyelidDetection
from Iris_Normalization import normalize_iris

def ImageEnhancement(normalized):
    enhanced = []
    for res in normalized:
        # Assicurati che l'immagine sia in scala di grigi
        if len(res.shape) == 3 and res.shape[2] == 3:
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        # Assicurati che l'immagine sia del tipo corretto
        res = res.astype(np.uint8)
        
        # Apply histogram equalization
        eq_hist = cv2.equalizeHist(res)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(eq_hist)

        # Resize the image to 64x512
        resized_img = cv2.resize(clahe_img, (512, 64))
        
        enhanced.append(resized_img)
    return enhanced


'''# Caricamento dell'immagine
image_path = 'data/CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp'  # Utilizza il percorso corretto dell'immagine
image = cv2.imread(image_path)

# Verifica se l'immagine è stata caricata correttamente
if image is None:
    print("L'immagine non è stata trovata o non può essere caricata.")
else:
    # Esecuzione di IrisLocalization
    img, boundaries, filtered_images = IrisLocalization([image])

    if not boundaries:
        print("No circles were found!")
    else:
        # Immagine con palpebra localizzata
        img, boundaries, upper_eyelid_curves, lower_eyelid_curves = EyelidDetection([img[0]], boundaries)

        # Normalizzazione dell'iride
        normalized_iris_imgs = normalize_iris([image], boundaries, upper_eyelid_curves, lower_eyelid_curves)
        normalized_iris = normalized_iris_imgs[0]

        # Esecuzione di ImageEnhancement
        enhanced_images = ImageEnhancement([normalized_iris])

        # Visualizzazione dei risultati
        plt.figure(figsize=(10, 5))

        # Immagine dell'iride normalizzata
        plt.subplot(1, 2, 1)
        plt.title('Normalized Iris')
        plt.imshow(normalized_iris, cmap='gray')
        plt.axis('off')

        # Immagine dell'iride migliorata
        enhanced_image = enhanced_images[0]
        plt.subplot(1, 2, 2)
        plt.title('Enhanced Iris')
        plt.imshow(enhanced_image, cmap='gray')
        plt.axis('off')

        plt.show()'''