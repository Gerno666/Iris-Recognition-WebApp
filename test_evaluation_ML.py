import os
from ML_test import ML_Match

def calculate_evaluation_metrics(correct_recognition_true, correct_recognition_false, false_positives, false_negatives, total_samples):
    # Calculation of the correct recognition rate (CRRT)
    CRRT = correct_recognition_true / (total_samples/108)

    # Calculation of the false negative rate (FNMR)
    FNMR = false_negatives / (total_samples/(108))

    # Calculation of the correct recognition rate (CRRF)
    CRRF = correct_recognition_false / (total_samples - (total_samples/(108)))

    # Calculation of the false positive rate (FMR)
    FMR = false_positives / (total_samples - (total_samples/(108)))

    return CRRT, CRRF, FMR, FNMR

# Path to the folder containing the dataset
dataset_folder = 'data/CASIA Iris Image Database (version 1.0)'

# Initialization of counters for calculating evaluation parameters
total_samples = 0
correct_recognition_false = 0
correct_recognition_true = 0
false_positives = 0
false_negatives = 0

# Iterate through all subfolders numbered 001 through 118
for person_folder in sorted(os.listdir(dataset_folder)):
    person_folder_path = os.path.join(dataset_folder, person_folder)

    #if (person_folder == '051'):
        #CRRT, CRRF, FMR, FNMR = calculate_evaluation_metrics(correct_recognition_true, correct_recognition_false, false_positives, false_negatives, total_samples)
        #print(CRRT, FNMR, CRRF, FMR, total_samples)
        #exit()
    #print(person_folder)
    
    # Iterating through subfolders '1' and '2'
    for eye_folder in ['1', '2']:
        eye_folder_path = os.path.join(person_folder_path, eye_folder)

        # Make sure the subfolder contains images
        if os.path.isdir(eye_folder_path):

            # Get the full path to all images in the subfolder
            image_files = [os.path.join(eye_folder_path, file) for file in os.listdir(eye_folder_path) if file.endswith('.bmp')]

            for reference_image_file in image_files:

                print(reference_image_file)

                if (reference_image_file != 'CASIA Iris Image Database (version 1.0)/095/1/095_1_1.bmp' and reference_image_file != 'CASIA Iris Image Database (version 1.0)/095/1/095_1_3.bmp'):
                    for person_folder_2 in sorted(os.listdir(dataset_folder)):
                        person_folder_path_2 = os.path.join(dataset_folder, person_folder_2)
                        
                        # Iterating through subfolders '1' and '2'
                        for eye_folder_2 in ['1', '2']:
                            eye_folder_path_2 = os.path.join(person_folder_path_2, eye_folder_2)

                            # Make sure the subfolder contains images
                            if os.path.isdir(eye_folder_path_2):

                                # Get the full path to all images in the subfolder
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


# Calculate the evaluation parameters
CRRT, CRRF, FMR, FNMR = calculate_evaluation_metrics(correct_recognition_true, correct_recognition_false, false_positives, false_negatives, total_samples)
print(CRRT, FNMR, CRRF, FMR, total_samples)

# 0.5918367346938775 0.40816326530612246 0.9864581346557314 0.013541865344268548 5292 (13.5)
# 0.7142857142857143 0.2857142857142857 0.9692923898531375 0.030707610146862484 5292 (13.8)
# 0.7142857142857143 0.2857142857142857 0.9593744039671943 0.04062559603280565 5292 (14.5)
# 0.7551020408163265 0.24489795918367346 0.9412550066755674 0.05874499332443257 10584 (14.5)

# 0.6402116402116402 0.35978835978835977 0.9643298648639103 0.03567013513608974 571536 (14.5)
# 0.7004081632653061 0.29959183673469386 0.9466374213236697 0.05336257867633035 571536 (14.8)