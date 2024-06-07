import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib


def apply_lda(feature_vector_train, feature_vector_test, components):
    '''TRAINING'''
    ft_train = feature_vector_train
    
    # get the classes of all training feature vectors
    y_train = []
    for i in range(0,108):
        for k in range(3):
            y_train.append(i + 1)
    y_train = np.array(y_train)
    
    # fit the LDA model on training data with n components
    sklearn_lda = LDA(n_components=components)
    sklearn_lda.fit(ft_train, y_train)

    # Save the trained model to disk
    joblib.dump(sklearn_lda, 'modello_LDA.pkl')
    
    # transform the training data
    red_train = sklearn_lda.transform(ft_train)
    
    '''TESTING'''
    ft_test = feature_vector_test
    
    # transform the testing data
    red_test = sklearn_lda.transform(ft_test)
    
    return red_train, red_test


def IrisMatching(features_test, features_train, components_list, masks_test=None, masks_train=None):

    # Calcolo delle matrici delle distanze
    distance_matrix_L1 = manhattan_distances(features_test, features_train)
    distance_matrix_L2 = euclidean_distances(features_test, features_train)
    distance_matrix_cosine = cosine_distances(features_test, features_train)

    distance_matrix_L1_comp = []
    distance_matrix_L2_comp = []
    distance_matrix_cosine_comp = []

    # Calcola le metriche con LDA per diversi numeri di componenti
    for components in components_list:
        print("Calcolo le metriche per componenti: ", components)
        red_train, red_test = apply_lda(features_train, features_test, components)

        distance_matrix_L1_c = manhattan_distances(red_test, red_train)
        distance_matrix_L2_c = euclidean_distances(red_test, red_train)
        distance_matrix_cosine_c = cosine_distances(red_test, red_train)

        distance_matrix_L1_comp.append(distance_matrix_L1_c)
        distance_matrix_L2_comp.append(distance_matrix_L2_c)
        distance_matrix_cosine_comp.append(distance_matrix_cosine_c)

    # Creazione delle etichette
    labels_train = [i for i in range(108) for _ in range(3)]  # 108 soggetti, 3 campioni per soggetto
    labels_test = [i for i in range(108) for _ in range(4)]  # 108 soggetti, 4 campioni per soggetto

    return (distance_matrix_L1, distance_matrix_L2, distance_matrix_cosine, distance_matrix_L1_comp, distance_matrix_L2_comp, distance_matrix_cosine_comp, labels_train, labels_test)
