import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# data without slacks
# X = np.array([[3,4],[1,4],[2,3],[6,-1],[7,-1],[5,-3]])
# y = np.array([-1,-1, -1, 1, 1, 1])

# data with slacks
# X = np.array([[3,4],[1,4],[2,3],[6,-1],[7,-1],[5,-3],[2,4], [4,4]])
# y = np.array([-1, -1, -1, 1, 1, 1, 1, 1])

# x_neg = X[0:3, ]
# x_pos = X[3:X.shape[0], ]

# data for regression
X = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


def svm_noSlack_dual(X,y):
    #Initializing values and computing H. Note the 1. to force to float type
    m,n = X.shape
    y = y.reshape(-1,1) * 1
    X_dash = y * X
    H = np.dot(X_dash, X_dash.T) * 1

    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    #Setting solver parameters
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    theta = ((y * alphas).T @ X).reshape(-1, 1)

    threshold = 1e-5
    S = (alphas > threshold).flatten()
    theta0 = y[S] - np.dot(X[S], theta)
    theta0 = theta0[0]

    # print results
    print('Alphas = ', alphas)
    print('theta = ', theta.flatten())
    print('theta0 = ', theta0)
    return theta, theta0


def svm_Slack_dual(X, y, C=1):
    #Initializing values and computing H. Note the 1. to force to float type
    m,n = X.shape
    y = y.reshape(-1, 1) * 1.
    X_dash = y * X
    H = np.dot(X_dash, X_dash.T) * 1.

    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    #Setting solver parameters
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    theta = ((y * alphas).T @ X).reshape(-1,1)

    threshold = 1e-5
    S = (alphas > threshold).flatten()
    theta0 = y[S] - np.dot(X[S], theta)
    theta0 = theta0[0]

    # print results
    print('Alphas = ', alphas)
    print('theta = ', theta.flatten())
    print('theta0 = ', theta0)
    return theta, theta0


def svm_regression(X,y,C=1):
    


def plot_decision_boundary(X, y, theta0, theta):
    x1 = np.array([min(X[:, 0]), max(X[:, 0])])
    m = -theta[0] / theta[1]
    c = -theta0 / theta[1]
    x2 = m * x1 + c

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(x_neg[:, 0], x_neg[:, 1], marker='x', color='r', label='Negative -1')
    plt.scatter(x_pos[:, 0], x_pos[:, 1], marker='o', color='b', label='Positive +1')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title('SVM')
    plt.plot(x1, x2, 'y-')


# results = svm_noSlack_dual(X,y)
# results = svm_Slack_dual(X,y,C=1)
# beta = results[0]
# beta0 = results[1]
# plot_decision_boundary(X, y, beta0, beta)



# # results = svm_noSlack_dual(X,y)
# results = svm_Slack_dual(X,y,C=1)
# beta = results[0]
# beta0 = results[1]
# plot_decision_boundary(X, y, beta0, beta)


# # sklearn
# from sklearn.svm import SVCdef
# clf = SVC(C = 10, kernel = 'linear')
# clf.fit(X, y.ravel())
# print(clf.coef_)