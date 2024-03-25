import os
from ML_test import ML_Match

def calculate_evaluation_metrics(correct_recognition_true, correct_recognition_false, false_positives, false_negatives, total_samples):
    # Calcolo del tasso di riconoscimento corretto (CRRT)
    CRRT = correct_recognition_true / (total_samples/108)

    # Calcolo del tasso di falsi negativi (FNMR)
    FNMR = false_negatives / (total_samples/(108))

    # Calcolo del tasso di riconoscimento corretto (CRRF)
    CRRF = correct_recognition_false / (total_samples - (total_samples/(108)))

    # Calcolo del tasso di falsi positivi (FMR)
    FMR = false_positives / (total_samples - (total_samples/(108)))

    return CRRT, CRRF, FMR, FNMR

# Percorso della cartella contenente il dataset
dataset_folder = 'data/CASIA Iris Image Database (version 1.0)'

# Inizializzazione dei contatori per il calcolo dei parametri di valutazione
total_samples = 0
correct_recognition_false = 0
correct_recognition_true = 0
false_positives = 0
false_negatives = 0

# Iterazione attraverso tutte le sottocartelle numerate da 001 a 118
for person_folder in sorted(os.listdir(dataset_folder)):
    person_folder_path = os.path.join(dataset_folder, person_folder)

    if (person_folder == '051'):
        CRRT, CRRF, FMR, FNMR = calculate_evaluation_metrics(correct_recognition_true, correct_recognition_false, false_positives, false_negatives, total_samples)
        print(CRRT, FNMR, CRRF, FMR, total_samples)
        exit()
    print(person_folder)
    
    # Iterazione attraverso le sottocartelle '1' e '2'
    for eye_folder in ['1', '2']:
        eye_folder_path = os.path.join(person_folder_path, eye_folder)

        # Assicurati che la sottocartella contenga le immagini
        if os.path.isdir(eye_folder_path):

            # Ottieni il percorso completo di tutte le immagini nella sottocartella
            image_files = [os.path.join(eye_folder_path, file) for file in os.listdir(eye_folder_path) if file.endswith('.bmp')]

            for reference_image_file in image_files:

                print(reference_image_file)

                if (reference_image_file != 'CASIA Iris Image Database (version 1.0)/095/1/095_1_1.bmp' and reference_image_file != 'CASIA Iris Image Database (version 1.0)/095/1/095_1_3.bmp'):
                    for person_folder_2 in sorted(os.listdir(dataset_folder)):
                        person_folder_path_2 = os.path.join(dataset_folder, person_folder_2)
                        
                        # Iterazione attraverso le sottocartelle '1' e '2'
                        for eye_folder_2 in ['1', '2']:
                            eye_folder_path_2 = os.path.join(person_folder_path_2, eye_folder_2)

                            # Assicurati che la sottocartella contenga le immagini
                            if os.path.isdir(eye_folder_path_2):

                                # Ottieni il percorso completo di tutte le immagini nella sottocartella
                                image_files_2 = [os.path.join(eye_folder_path_2, file_2) for file_2 in os.listdir(eye_folder_path_2) if file_2.endswith('.bmp')]

                                for reference_image_file_2 in image_files_2:
                    
                                    bool = ML_Match(reference_image_file, reference_image_file_2)
                                    total_samples+=1
                                    
                                    if (bool):
                                        if (person_folder == person_folder_2):
                                            correct_recognition_true += 1
                                        else:
                                            false_positives += 1
                                    else:
                                        if (person_folder == person_folder_2):
                                            false_negatives += 1
                                        else:
                                            correct_recognition_false += 1


# Calcola i parametri di valutazione
CRRT, CRRF, FMR, FNMR = calculate_evaluation_metrics(correct_recognition_true, correct_recognition_false, false_positives, false_negatives, total_samples)
print(CRRT, FNMR, CRRF, FMR, total_samples)

# 0.5918367346938775 0.40816326530612246 0.9864581346557314 0.013541865344268548 5292 (13.5)
# 0.7142857142857143 0.2857142857142857 0.9692923898531375 0.030707610146862484 5292 (13.8)
# 0.7142857142857143 0.2857142857142857 0.9593744039671943 0.04062559603280565 5292 (14.5)
# 0.7551020408163265 0.24489795918367346 0.9412550066755674 0.05874499332443257 10584 (14.5)

# 0.6402116402116402 0.35978835978835977 0.9643298648639103 0.03567013513608974 571536 (14.5)
# 0.6693121693121693 0.3306878306878307 0.9554891530859488 0.04451084691405119 571536 (14.8)