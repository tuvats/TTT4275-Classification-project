import numpy as np

# class LogisticRegression:
#     def __init__(self, learning_rate=0.5, epochs=1000):
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.weights, self.bias = None, None
#         self.losses, self.train_accuacies = [], []

#     def sigmoid_function(self, X):
#         return 1/(1+np.exp(-X))
    
#     def mean_square_error(self, X, T):
#         return np.mean((np.transpose(X-T))*(X-T))
    

def sigmoid_function(z):
    return 1/(1+np.exp(-z))

def train_linear_classifier(X, T, alpha=0.5, max_iter=1000, tol=1e-5):
    N, D = X.shape
    C = T.shape[1]
    W = np.zeros((C, D))
    mse_history = []

    for epoch in range(max_iter):
        Z = X @ W.T
        G = sigmoid_function(Z)
        E = G - T
        mse = np.mean(np.sum(E**2, axis=1))
        mse_history.append(mse)

        grad = (E*G*(1-G)).T@X
        W -= alpha*grad

    return W, mse_history

def predict(X, W):
    lin_model = W@X