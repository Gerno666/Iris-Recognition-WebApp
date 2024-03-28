import numpy as np
import cv2
import math
import joblib
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction

# Path of the trained model
model_path = 'modello_LDA.pkl'
# Load the trained model
sklearn_lda = joblib.load(model_path)


def extract_features(image):
    image = cv2.imread(image)
    image = [image]
    # Extract vector features from the image
    boundary,centers=IrisLocalization(image)
    normalized=IrisNormalization(boundary,centers)
    enhanced=ImageEnhancement(normalized)
    feature_vector=FeatureExtraction(enhanced)
    return feature_vector

def perform_matching(feature_vector1, feature_vector2):
    feature_vector1_np = np.array(feature_vector1)
    feature_vector2_np = np.array(feature_vector2)

    if feature_vector1_np.shape[0] != 1:
        feature_vector1_np = feature_vector1_np.reshape(1, -1)

    if feature_vector2_np.shape[0] != 1:
        feature_vector2_np = feature_vector2_np.reshape(1, -1)

    # Transform feature vectors using the trained LDA model
    feature1_transformed = sklearn_lda.transform(feature_vector1_np)
    feature2_transformed = sklearn_lda.transform(feature_vector2_np)
    
    # Calculate the Euclidean distance between the transformed features
    distance = np.linalg.norm(feature1_transformed - feature2_transformed)
    
    # Define a threshold to determine whether the two images match or not
    threshold = 14.8 
    
    # Determine if the two images match using distance and threshold
    if distance <= threshold:
        match = True
    else:
        match = False
    
    return match

def ML_Match(path1, path2):
    # Extract vector features from images
    feature_vector1 = extract_features(path1)
    feature_vector2 = extract_features(path2)

    # Match the two images using vector features and the trained model
    match = perform_matching(feature_vector1, feature_vector2)

    if match:
        return True
    else:
        return False