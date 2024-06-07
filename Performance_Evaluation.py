import numpy as np

def PerformanceEvaluation(distance_matrix, labels_train, labels_test, ranks=432):
    num_probes, num_gallery = distance_matrix.shape
    cms_k = [0] * ranks
    
    for i in range(num_probes):
        sorted_indices = np.argsort(distance_matrix[i])
        
        for k in range(num_gallery):
            if sorted_indices[k] < len(labels_train) and i < len(labels_test) and labels_train[sorted_indices[k]] == labels_test[i]:
                if k < ranks:
                    cms_k[k] += 1
                break
    
    # Converti CMS(k) in percentuali e rendi cumulativo
    cms_k = [100 * (x / num_probes) for x in cms_k]
    for i in range(1, len(cms_k)):
        cms_k[i] += cms_k[i - 1]
    
    cmc = cms_k.copy()
    rr = cms_k[0]
    
    return cms_k, cmc, rr