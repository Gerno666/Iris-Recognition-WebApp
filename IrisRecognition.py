# coding: utf-8

import cv2
import numpy as np
import glob
import math
import scipy
from scipy.spatial import distance
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from Iris_Localization import IrisLocalization
from Eyelid_Detection import EyelidDetection
from Iris_Normalization import normalize_iris
from Image_Enhancement import ImageEnhancement
from Feature_Extraction import FeatureExtraction
from Iris_Matching import IrisMatching
from Performance_Evaluation import PerformanceEvaluation
import warnings
warnings.filterwarnings("ignore")

'''TRAINING'''

# reading the training images from the CASIA dataset
images_train = [cv2.imread(file) for file in sorted(glob.glob('data/CASIA Iris Image Database (version 1.0)/*/1/*.bmp'))]
num_data_train = len(images_train)
print("Numero di dati di training:", num_data_train)

# running Localization, Normalization,Enhancement and Feature Extraction on all the training images
images_train, boundary_train, centers_train,  = IrisLocalization(images_train)
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
images_test, boundary_test, centers_test,  = IrisLocalization(images_test)
images_test, boundary_test, upper_eyelid_curves_test, lower_eyelid_curves_test = EyelidDetection(images_test, boundary_test)
normalized_test = normalize_iris(images_test, boundary_test, upper_eyelid_curves_test, lower_eyelid_curves_test)
enhanced_test = ImageEnhancement(normalized_test)
feature_vector_test = FeatureExtraction(enhanced_test)
print("Testing data processed.")

# Lista dei numeri di componenti per LDA
components_list = [10, 30, 50, 70, 90, 107]

distance_matrix_L1, distance_matrix_L2, distance_matrix_cosine, distance_matrix_L1_comp, distance_matrix_L2_comp, distance_matrix_cosine_comp, labels_train, labels_test = IrisMatching(feature_vector_test, feature_vector_train, components_list)

# Calcola le metriche per L1
cms_k_L1, cmc_L1, rr_L1 = PerformanceEvaluation(distance_matrix_L1, labels_train, labels_test)

# Calcola le metriche per L2
cms_k_L2, cmc_L2, rr_L2 = PerformanceEvaluation(distance_matrix_L2, labels_train, labels_test)

# Calcola le metriche per Cosine
cms_k_cosine, cmc_cosine, rr_cosine = PerformanceEvaluation(distance_matrix_cosine, labels_train, labels_test)


def plot_cmc(cmc, label):
    plt.plot(range(1, len(cmc) + 1), cmc, label=label)

# Stampa le metriche
print("L1 Metrics:")
print("Recognition Rate (RR):", rr_L1)

print("\nL2 Metrics:")
print("Recognition Rate (RR):", rr_L2)

print("\nCosine Metrics:")
print("Recognition Rate (RR):", rr_cosine)

# Plotta le CMC curve
plt.figure()
plot_cmc(cmc_L1, label='L1')
plot_cmc(cmc_L2, label='L2')
plot_cmc(cmc_cosine, label='Cosine')

# Aggiungi etichette e titolo
plt.title('Cumulative Match Characteristic (CMC) Curve')
plt.xlabel('Rank (k)')
plt.ylabel('CMS(k)')
plt.legend()
plt.grid(True)

# Mostra il grafico
plt.show()


for i, components in zip(range(6), components_list):
    cms_k_L1, cmc_L1, rr_L1 = PerformanceEvaluation(distance_matrix_L1_comp[i], labels_train, labels_test)
    cms_k_L2, cmc_L2, rr_L2 = PerformanceEvaluation(distance_matrix_L2_comp[i], labels_train, labels_test)
    cms_k_cosine, cmc_cosine, rr_cosine = PerformanceEvaluation(distance_matrix_cosine_comp[i], labels_train, labels_test)

    # Stampa le metriche con LDA
    print(f"\nWith LDA (components={components}):")
    print("L1 Metrics:")
    print("Recognition Rate (RR):", rr_L1)

    print("\nL2 Metrics:")
    print("Recognition Rate (RR):", rr_L2)

    print("\nCosine Metrics:")
    print("Recognition Rate (RR):", rr_cosine)

    # Plotta le CMC curve con LDA
    plot_cmc(cmc_L1, label=f'L1 with LDA (components={components})')
    plot_cmc(cmc_L2, label=f'L2 with LDA (components={components})')
    plot_cmc(cmc_cosine, label=f'Cosine with LDA (components={components})')

    # Aggiungi etichette e titolo
    plt.title('Cumulative Match Characteristic (CMC) Curve')
    plt.xlabel('Rank (k)')
    plt.ylabel('CMS(k)')
    plt.legend()
    plt.grid(True)

    # Mostra il grafico
    plt.show()
