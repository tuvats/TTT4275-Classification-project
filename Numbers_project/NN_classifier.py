import numpy as np
import time
from scipy.spatial.distance import cdist

def get_euclidian_distance(img1, img2):
    x = img1.flatten() #Turns vector of shape (28, 28) into vector of length 784
    mu = img2.flatten()
    diff = x-mu
    return np.dot(diff, diff)

def NN_classifier(train_data, train_labels, test_data, test_labels):
    start = time.time() #Tracking runtime
    num_train = len(train_data)
    num_test = len(test_data)
    conf_matrix = np.zeros((10, 10), dtype=int)
    correct_pred = []
    wrong_pred = []

    for i in range(num_test):
        test_img = test_data[i]
        min_d = float('inf') # Setting minimum distance to a very large value
        predicted_label = None

        for j in range(num_train):
            train_img = train_data[j]
            d = get_euclidian_distance(test_img, train_img)

            if d < min_d:
                min_d = d
                predicted_label = train_labels[j]
        if predicted_label == test_labels[i]:
            correct_pred.append(test_img)
            conf_matrix[test_labels[i], test_labels[i]] += 1
        else:
            wrong_pred.append(test_img)
            conf_matrix[test_labels[i], predicted_label] += 1
    
    end = time.time()
    print(f"Runtime of program is {end - start:.2f} seconds.")
    return conf_matrix, correct_pred, wrong_pred

def NN_classifier_new(train_data, train_labels, test_data, test_labels):
    start = time.time()

    train_data_flat = train_data.reshape(len(train_data), -1)
    test_data_flat = test_data.reshape(len(test_data), -1)


    distance = cdist(test_data_flat, train_data_flat, metric='euclidean')

    nearest_idx = np.argmin(distance, axis=1)
    predicted_labels = train_labels[nearest_idx]

    # Initialize confusion matrix
    conf_matrix = np.zeros((10, 10), dtype=int)
    correct_pred = []
    wrong_pred = []
    
    # Fill confusion matrix and find correct/incorrect predictions
    for i, (true_label, pred_label) in enumerate(zip(test_labels, predicted_labels)):
        conf_matrix[true_label, pred_label] += 1
        if true_label == pred_label:
            correct_pred.append(test_data[i])
        else:
            wrong_pred.append(test_data[i])

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds")
    return conf_matrix, correct_pred, wrong_pred