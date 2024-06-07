import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib
from Iris_Localization import IrisLocalization
from Eyelid_Detection import EyelidDetection
from Iris_Normalization import normalize_iris
from Image_Enhancement import ImageEnhancement
from Feature_Extraction import FeatureExtraction


# Applicazione di LDA
def apply_lda(feature_vector_train, feature_vector_test, components):
    ft_train = feature_vector_train
    y_train = [i for i in range(108) for _ in range(3)]
    y_train = np.array(y_train)

    lda = LDA(n_components=components)
    lda.fit(ft_train, y_train)
    joblib.dump(lda, 'modello_LDA.pkl')

    red_train = lda.transform(ft_train)
    red_test = lda.transform(feature_vector_test)
    
    return red_train, red_test

# Calcolo delle metriche di verifica
def verify(distances, labels_train, labels_test, threshold):
    GA, FR, FA, GR = 0, 0, 0, 0
    total_genuine = 0
    total_impostor = 0
    
    for i, test_label in enumerate(labels_test):
        for j, train_label in enumerate(labels_train):
            distance = distances[i, j]
            is_genuine = (test_label == train_label)
            if is_genuine:
                total_genuine += 1
            else:
                total_impostor += 1
            
            if distance <= threshold:
                if is_genuine:
                    GA += 1
                else:
                    FA += 1
            else:
                if is_genuine:
                    FR += 1
                else:
                    GR += 1

    return GA, FR, FA, GR, total_genuine, total_impostor

# Calcolo delle metriche FAR, FRR, GAR, GRR
def calculate_metrics(GA, FR, FA, GR, total_genuine, total_impostor):
    FAR = FA / total_impostor if total_impostor else 0
    FRR = FR / total_genuine if total_genuine else 0
    GAR = 1 - FRR
    GRR = 1 - FAR
    return FAR, FRR, GAR, GRR

# Calcolo della matrice delle distanze con la metrica Coseno
def calculate_distances(red_train, red_test):
    distances = cosine_distances(red_test, red_train)
    return distances

# Tracciare la curva ROC
def plot_roc(FARs, GARs, EER_index):
    plt.plot(FARs, GARs, marker='o', label='ROC Curve')
    plt.scatter(FARs[EER_index], GARs[EER_index], color='red', zorder=5)  # Evidenzia EER in rosso
    plt.text(FARs[EER_index], GARs[EER_index], 'EER', fontsize=12, color='red', ha='right')
    plt.title('ROC Curve')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('Genuine Acceptance Rate (GAR)')
    plt.legend()
    plt.grid(True)
    plt.show()

'''TRAINING'''

# reading the training images from the CASIA dataset
images_train = [cv2.imread(file) for file in sorted(glob.glob('data/CASIA Iris Image Database (version 1.0)/*/1/*.bmp'))]
num_data_train = len(images_train)
print("Numero di dati di training:", num_data_train)

# running Localization, Normalization,Enhancement and Feature Extraction on all the training images
images_train, boundary_train, _  = IrisLocalization(images_train)
images_train, boundary_train, upper_eyelid_curves_train, lower_eyelid_curves_train = EyelidDetection(images_train, boundary_train)
normalized_train = normalize_iris(images_train, boundary_train, upper_eyelid_curves_train, lower_eyelid_curves_train)
enhanced_train = ImageEnhancement(normalized_train)
feature_vector_train = FeatureExtraction(enhanced_train)
print("Training data processed.")


'''TESTING'''

# reading the testing images from the CASIA dataset
images_test = [cv2.imread(file) for file in sorted(glob.glob('data/CASIA Iris Image Database (version 1.0)/*/2/*.bmp'))]
num_data_test = len(images_test)
print("Numero di dati di testing:", num_data_test)

# running Localization, Normalization,Enhancement and Feature Extraction on all the testing images
images_test, boundary_test, _  = IrisLocalization(images_test)
images_test, boundary_test, upper_eyelid_curves_test, lower_eyelid_curves_test = EyelidDetection(images_test, boundary_test)
normalized_test = normalize_iris(images_test, boundary_test, upper_eyelid_curves_test, lower_eyelid_curves_test)
enhanced_test = ImageEnhancement(normalized_test)
feature_vector_test = FeatureExtraction(enhanced_test)
print("Testing data processed.")

# Applicazione di LDA
components = 107
red_train, red_test = apply_lda(feature_vector_train, feature_vector_test, components)

# Calcolo della matrice delle distanze con la metrica Coseno
distance_matrix = calculate_distances(red_train, red_test)

# Creazione delle etichette
labels_train = [i for i in range(108) for _ in range(3)]  # 108 soggetti, 3 campioni per soggetto
labels_test = [i for i in range(108) for _ in range(4)]  # 108 soggetti, 4 campioni per soggetto

# Definizione della soglia
thresholds = np.linspace(np.min(distance_matrix), np.max(distance_matrix), 100)
FARs, FRRs, GARs, GRRs = [], [], [], []

for threshold in thresholds:
    GA, FR, FA, GR, total_genuine, total_impostor = verify(distance_matrix, labels_train, labels_test, threshold)
    FAR, FRR, GAR, GRR = calculate_metrics(GA, FR, FA, GR, total_genuine, total_impostor)
    FARs.append(FAR)
    FRRs.append(FRR)
    GARs.append(GAR)
    GRRs.append(GRR)

# Calcolo di EER
EER_threshold_index = np.argmin(np.abs(np.array(FARs) - np.array(FRRs)))
EER = (FARs[EER_threshold_index] + FRRs[EER_threshold_index]) / 2

print(f"EER: {EER:.4f}")
print(f"EER treshold: {thresholds[EER_threshold_index]:.4f}")

# Plot della curva ROC
plot_roc(FARs, GARs, EER_threshold_index)

# Calcolo delle metriche finali per il punto EER
final_FAR = FARs[EER_threshold_index]*100
final_FRR = FRRs[EER_threshold_index]*100
final_GAR = GARs[EER_threshold_index]*100
final_GRR = GRRs[EER_threshold_index]*100

# Stampare le metriche finali per il punto EER
print(f"Final FAR (EER): {final_FAR:.4f}, FRR (EER): {final_FRR:.4f}, GAR (EER): {final_GAR:.4f}, GRR (EER): {final_GRR:.4f}")
