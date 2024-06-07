import numpy as np
import scipy.signal
import cv2
import matplotlib.pyplot as plt
from Iris_Localization import IrisLocalization
from Eyelid_Detection import EyelidDetection
from Iris_Normalization import normalize_iris
from Image_Enhancement import ImageEnhancement

def m(x, y, f):
    return np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))

def gabor(x, y, dx, dy, f):
    gb = (1 / (2 * np.pi * dx * dy)) * np.exp(-0.5 * (x**2 / dx**2 + y**2 / dy**2)) * m(x, y, f)
    return gb

# Function to calculate spatial filter over 8x8 blocks
def spatial(f, dx, dy):
    sfilter = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            sfilter[i, j] = gabor(-4 + j, -4 + i, dx, dy, f)
    return sfilter

def get_vec(convolved_train1, convolved_train2):
    feature_vec = []
    for i in range(6):
        for j in range(64):
            # Run 8 by 8 filtered block iteratively over the entire image
            start_height = i * 8
            end_height = start_height + 8
            start_width = j * 8
            end_width = start_width + 8
            grid1 = convolved_train1[start_height:end_height, start_width:end_width]
            grid2 = convolved_train2[start_height:end_height, start_width:end_width]

            # Channel 1
            abs_grid1 = np.absolute(grid1)
            mean1 = np.mean(abs_grid1)
            std1 = np.mean(np.absolute(abs_grid1 - mean1))
            feature_vec.extend([mean1, std1])

            # Channel 2
            abs_grid2 = np.absolute(grid2)
            mean2 = np.mean(abs_grid2)
            std2 = np.mean(np.absolute(abs_grid2 - mean2))
            feature_vec.extend([mean2, std2])

    return feature_vec

def FeatureExtraction(enhanced):
    # Get spatial filters
    filter1 = spatial(0.67, 3, 1.5)
    filter2 = spatial(0.67, 4, 1.5)
    
    feature_vectors = []

    for img in enhanced:
        # Define a 48x512 region over which the filters are applied
        img_roi = img[:48, :]

        filtered1 = scipy.signal.convolve2d(img_roi, filter1, mode='same')
        filtered2 = scipy.signal.convolve2d(img_roi, filter2, mode='same')
        
        fv = get_vec(filtered1, filtered2)
        feature_vectors.append(fv)

    return feature_vectors  # Each feature vector has a dimension of 1536


'''# Esempio di Visualizzazione delle Feature
def visualize_features(features):
    for i, feature in enumerate(features):
        # Convert the feature vector to a 2D array for visualization
        feature_matrix = np.array(feature).reshape(-1, 32)
        
        plt.figure(figsize=(10, 5))
        plt.title(f'Feature Map {i+1}')
        plt.imshow(feature_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Feature Value')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Index')
        plt.show()


image_path = 'data/CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp'  # Utilizza il percorso corretto dell'immagine
image = cv2.imread(image_path)

# Verifica se l'immagine è stata caricata correttamente
if image is None:
    print("L'immagine non è stata trovata o non può essere caricata.")
else:
    # Esecuzione di IrisLocalization
    img, boundary, filtered_images = IrisLocalization([image])


    if not boundary:
        print("No circles were found!")
    else:
        # Immagine con palpebra localizzata
        img, boundaries, upper_eyelid_curves, lower_eyelid_curves = EyelidDetection([img[0]], boundary)

        # Normalizzazione dell'iride
        normalized_iris_imgs = normalize_iris([image], boundaries, upper_eyelid_curves, lower_eyelid_curves)
        normalized_iris = normalized_iris_imgs[0]

        # Esecuzione di ImageEnhancement
        enhanced_images = ImageEnhancement(normalized_iris)

        # Esecuzione di FeatureExtraction
        features = FeatureExtraction(enhanced_images)
        visualize_features(features)'''