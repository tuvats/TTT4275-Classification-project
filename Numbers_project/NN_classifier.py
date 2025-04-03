import numpy as np
import time
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def NN_classifier(train_data, train_labels, test_data, test_labels):
    start = time.time()

    train_data_flat = train_data.reshape(len(train_data), -1)
    test_data_flat = test_data.reshape(len(test_data), -1)

    distance = cdist(test_data_flat, train_data_flat, metric='euclidean')

    nearest_idx = np.argmin(distance, axis=1)
    predicted_labels = train_labels[nearest_idx]

    # Initialize confusion matrix
    conf_matrix = np.zeros((10, 10), dtype=int)
    correct_pred = []
    correct_labels = []
    wrong_pred = []
    wrong_labels = []
    
    # Fill confusion matrix and find correct/incorrect predictions
    for i, (true_label, pred_label) in enumerate(zip(test_labels, predicted_labels)):
        conf_matrix[true_label, pred_label] += 1
        if true_label == pred_label:
            correct_pred.append(test_data[i])
            correct_labels.append(true_label)
        else:
            wrong_pred.append(test_data[i])
            wrong_labels.append((true_label, pred_label))

    num_wrong = len(wrong_pred)
    num_total = len(test_labels)
    error_rate = (num_wrong/num_total)*100

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds")
    print(f"Error Rate: {error_rate:.2f}%")
    return conf_matrix, correct_pred, wrong_pred, correct_labels, wrong_labels


def clustering(data, labels, n_clusters=64):
    n_classes = 10
    data_flat = data.reshape(len(data), -1) #Flatten images
    all_centers = [] # Array of shape [n_classes * n_clusters, 784]
    all_labels = [] # Array of shape [n_classes * n_clusters]

    for digit in range(n_classes):
        class_data = data_flat[digit==labels]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(class_data)
        centers = kmeans.cluster_centers_

        all_centers.append(centers)
        all_labels.extend([digit]*64)

    all_centers = np.vstack(all_centers)
    all_labels = np.array(all_labels)

    return all_centers, all_labels


def k_NN_classifier(train_data, train_labels, test_data, test_labels, k=7):
    start = time.time()

    train_data_flat = train_data.reshape(len(train_data), -1)
    test_data_flat = test_data.reshape(len(test_data), -1)

    distance = cdist(test_data_flat, train_data_flat, metric='euclidean')
    nearest_idx = np.argsort(distance, axis=1)[:, :k]

    # Predicting using majority vote
    predicted_labels = []
    for idx in nearest_idx:
        nearest_labels = train_labels[idx]
        values, counts = np.unique(nearest_labels, return_counts=True)
        # values get unique label values (ex: [3, 7]), and counts get how many times each appears (ex: [2, 1] if 3 appears 2 times and 7 once)
        majority_label = values[np.argmax(counts)]
        # np.argmax(counts) returns index of largest count, and values[] gives label with largest count
        predicted_labels.append(majority_label)
    
    predicted_labels = np.array(predicted_labels)

    conf_matrix = np.zeros((10, 10), dtype=int)
    correct_pred = []
    correct_labels = []
    wrong_pred = []
    wrong_labels = []

    for i, (true_label, pred_label) in enumerate(zip(test_labels, predicted_labels)):
        true_label = int(true_label)
        pred_label = int(pred_label)

        conf_matrix[true_label, pred_label] += 1

        if true_label == pred_label:
            correct_pred.append(test_data[i])
            correct_labels.append(true_label)
        else:
            wrong_pred.append(test_data[i])
            wrong_labels.append((true_label, pred_label))

    num_wrong = len(wrong_pred)
    num_total = len(test_labels)
    error_rate = (num_wrong/num_total)*100

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds")
    print(f"Error Rate: {error_rate:.2f}%")

    return conf_matrix, correct_pred, wrong_pred, correct_labels, wrong_labels