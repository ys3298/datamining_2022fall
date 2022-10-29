from sklearn import datasets
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
X, y = datasets.make_blobs(n_samples=20,n_features=2,centers=2,cluster_std=1)
y[y==0]=-1

def pred(X_inp, beta_inp, beta0_inp):
    # beta_inp is a vector of size X_inp.shape[1]
    # beta0_inp is a scalar
    a = np.matmul(X_inp, beta_inp) + beta0_inp
    y_pred = np.repeat(-1, X_inp.shape[0])
    for i in range(len(a)):
        if a[i] >= 0:
            y_pred[i] = 1
    return y_pred


def perceptron_nonStandard(X_train, y_train, beta, beta0, rho, mis_class = 5):
    beta_old = np.copy(beta)
    beta0_old = np.copy(beta0)
    ite = 1
    while ite == 1 or len(mis) > mis_class:
        ite = ite + 1
        beta_old = np.copy(beta)
        beta0_old = np.copy(beta0)

        pred_y = pred(X_train, beta, beta0)
        mis = np.asarray(np.where(pred_y != y_train.reshape(-1))).reshape(-1)
        if len(mis) != 0:
            one_ind = random.choice(mis)
            beta = beta_old + rho * X_train[one_ind,:]*y_train[one_ind]
            beta0 = beta0_old + rho * y_train[one_ind]

        print(len(mis))
    return np.insert(beta_old, 0, beta0_old)


def prediction_accuracy(estimated_beta, y_true, X_true):
    predicted = pred(X_true, estimated_beta[1:len(estimated_beta)], estimated_beta[0])
    acc = sum(predicted == y_true)/len(predicted)
    print(acc)
    return acc,predicted


beta_hat = perceptron_nonStandard(X, y, beta=np.ones(X.shape[1]), beta0=0, rho=2, mis_class = 0)
results = prediction_accuracy(beta_hat, y, X)


# visulize the results
# reference: https://towardsdatascience.com/exploring-the-perceptron-algorithm-using-python-c1d3af53a7c7


def plot_decision_boundary(X, theta):
    # X --> Inputs
    # theta --> parameters

    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = np.array([min(X[:, 0]), max(X[:, 0])])
    m = -theta[1] / theta[2]
    c = -theta[0] / theta[2]
    x2 = m * x1 + c

    fig = plt.figure(figsize=(10, 8))
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], "r^")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title('Perceptron Algorithm')
    plt.plot(x1, x2, 'y-')


plot_decision_boundary(X, beta_hat)