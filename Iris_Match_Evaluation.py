import glob
import os
from tqdm import tqdm
from Iris_Match import match_images

# Cartella delle immagini test e di training
test_folder = 'data/CASIA Iris Image Database (version 1.0)/*/2/*.bmp'
train_folder = 'data/CASIA Iris Image Database (version 1.0)/*/1/*.bmp'

# Inizializza le variabili per contare i match e le comparazioni totali
num_genuine_matches = 0
num_impostor_matches = 0
total_genuine_comparisons = 0
total_impostor_comparisons = 0

# Utilizza tqdm per creare una barra di avanzamento
for test_path in tqdm(sorted(glob.glob(test_folder)), desc="Processing images"):

    # Dividi il percorso in componenti
    path_components = test_path.split(os.path.sep)
    # Ottieni il numero di soggetto
    test_subject = int(path_components[-3])  # L'indice -3 per ottenere il numero soggetto

    print (test_subject)

    if test_subject < 10:

        for train_path in sorted(glob.glob(train_folder)):

            # Dividi il percorso in componenti
            path_components = train_path.split(os.path.sep)
            # Ottieni il numero di soggetto
            train_subject = int(path_components[-3])  # L'indice -3 per ottenere il numero soggetto

            # Effettua il match tra le due immagini
            match_result = match_images(test_path, train_path)
            
            # Verifica se i soggetti corrispondono
            if test_subject == train_subject:
                # Le immagini appartengono allo stesso soggetto, quindi sono genuine
                total_genuine_comparisons += 1
                if match_images(test_path, train_path):
                    num_genuine_matches += 1
            else:
                # Le immagini appartengono a soggetti diversi, quindi sono impostor
                total_impostor_comparisons += 1
                if not match_images(test_path, train_path):
                    num_impostor_matches += 1

# Calcola FAR, FRR, GAR e GRR
GAR = num_genuine_matches / total_genuine_comparisons
GRR = num_impostor_matches / total_impostor_comparisons
FAR = 1 - GRR
FRR = 1 - GAR

# Stampa i risultati
print(f"FAR: {FAR}")
print(f"FRR: {FRR}")
print(f"GAR: {GAR}")
print(f"GRR: {GRR}")

'''
treshold: 0.725
FAR: 0.01
FRR: 0.366
GAR: 0.634
GRR: 0.99
'''

