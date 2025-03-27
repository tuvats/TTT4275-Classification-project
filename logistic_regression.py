import numpy as np

def sigmoid_function(z):
    return 1/(1+np.exp(-z))

def train_linear_classifier(X, T, alpha=0.01, max_iter=1000, tol=1e-5):
    N, D = X.shape
    C = T.shape[1]
    W = np.zeros((C, D))
    mse_history = []
    error_rates = []
    confusion_matrices = []

    for epoch in range(max_iter):
        Z = X @ W.T
        G = sigmoid_function(Z)
        E = G - T
        y_pred = np.argmax(G, axis=1)
        mse = np.mean(np.sum(E**2, axis=1))
        mse_history.append(mse)
        y_true = np.argmax(T, axis=1)
        error_rate = np.sum(y_true != y_pred) / len(y_true)
        error_rates.append(error_rate)
        
        #Updating the weights
        grad_g_MSE = G - T
        grad_z_g = G * (1-G)
        grad_W_z = []
        grad_W_z = np.array([x.T for x in X])
        grad_W_MSE = np.zeros((C, D))
        for c in range(C):
            for d in range(D):
                grad_W_MSE[c, d] = np.sum(grad_g_MSE[:, c] * grad_z_g[:, c] * grad_W_z[:, d])
        W -= alpha*grad_W_MSE
        
    return W, mse_history, error_rates

def predict(X, W):
    lin_model = X @ W.T
    G = sigmoid_function(lin_model)
    preds = np.argmax(G, axis=1)
    return preds

def confusion_matrix(y_pred, y_true):
    return 