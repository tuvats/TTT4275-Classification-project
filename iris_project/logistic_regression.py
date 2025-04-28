import numpy as np
from confusion_matrix import get_confusion_matrix

def sigmoid_function(z):
    return 1/(1+np.exp(-z))

def get_error_rate(y_true, y_pred):
    return np.sum(y_true != y_pred) / len(y_true)

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def train_linear_classifier(X, labels, alpha=0.01, max_iter=1000):
    N, D = X.shape
    X_aug = np.hstack([X, np.ones((N, 1))])  # (N, D+1)
    W = np.zeros((3, D+1))
    mse_history = []
    error_rates = []
    confusion_matrices = []
    train_accuracies = []
    t_k = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    for epoch in range(max_iter):
        Z = W @ X_aug.T
        G = sigmoid_function(Z)
        y_pred = np.argmax(G, axis=0)
        y_true = labels
        y_true = np.argmax(y_true, axis=1)  # Convert one-hot to label indices
        error_rate = np.sum(y_true != y_pred) / len(y_true)
        error_rates.append(error_rate)
        accuracy = compute_accuracy(y_true, y_pred)
        train_accuracies.append(accuracy)
        cm = get_confusion_matrix(y_true, y_pred)
        confusion_matrices.append(cm)
        mse = 0
        grad_MSE = np.zeros((3, X_aug.shape[1]))

        #Updating the weights
        for i in range(G.shape[1]):
            grad_G_MSE = G[:, i] - t_k[:, int(y_true[i])]
            grad_MSE += np.dot((grad_G_MSE*G[:, i]*(1-G[:, i])).reshape(3, 1), X_aug.T[:, i].reshape(1, D+1))
            mse += np.dot(grad_G_MSE.T, grad_G_MSE)

        mse_history.append(mse/2)
        W -= alpha*grad_MSE

    return W, mse_history, error_rates, train_accuracies, confusion_matrices

def predict(X, W):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    lin_model = np.dot(W, X_aug.T)
    preds = np.argmax(lin_model, axis=0)
    return preds