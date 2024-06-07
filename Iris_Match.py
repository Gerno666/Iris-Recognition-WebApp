import cv2
from Iris_Localization import IrisLocalization
from Eyelid_Detection import EyelidDetection
from Iris_Normalization import normalize_iris
from Image_Enhancement import ImageEnhancement
from Feature_Extraction import FeatureExtraction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_distances
import joblib

def match_images(image_path1, image_path2, threshold=0.725):
    # Caricare le immagini
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    # Eseguire le operazioni di pre-processing su entrambe le immagini
    image1, boundary1, _ = IrisLocalization([image1])
    image1, boundary1, upper_eyelid_curves1, lower_eyelid_curves1 = EyelidDetection(image1, boundary1)
    normalized_image1 = normalize_iris(image1, boundary1, upper_eyelid_curves1, lower_eyelid_curves1)
    enhanced_image1 = ImageEnhancement(normalized_image1)
    feature_vector1 = FeatureExtraction(enhanced_image1)
    
    image2, boundary2, _  = IrisLocalization([image2])
    image2, boundary2, upper_eyelid_curves2, lower_eyelid_curves2 = EyelidDetection(image2, boundary2)
    normalized_image2 = normalize_iris(image2, boundary2, upper_eyelid_curves2, lower_eyelid_curves2)
    enhanced_image2 = ImageEnhancement(normalized_image2)
    feature_vector2 = FeatureExtraction(enhanced_image2)
    
    # Applicare LDA se necessario
    lda = joblib.load('modello_LDA.pkl')  # Carica il modello LDA addestrato in precedenza
    red_vector1 = lda.transform(feature_vector1)
    red_vector2 = lda.transform(feature_vector2)
    
    # Calcolare la distanza tra le due immagini utilizzando la metrica desiderata
    distance = cosine_distances(red_vector1, red_vector2)[0][0]
    
    # Confrontare la distanza con la soglia di accettazione
    if distance <= threshold:
        return True
    else:
        return False