from sklearn.datasets import make_blobs
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cvxpy as cp

X_noSlack, labels_noSlack = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.25,  random_state=1234)
X_Slack, labels_Slack = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2,  random_state=1234)

# plt.figure(figsize=(5, 5))
# plt.title('Two blobs')
# plt.scatter(X_Slack[:, 0], X_Slack[:, 1], c=labels_Slack, s=25);


def svm_noSlack(X, labels):
    # split the data in the two classes. Name them class_1 and class_2.
    ## Assign label 0 to class_1
    class_1 = X[labels == 0,]
    ## Assign label 1 to class_2
    class_2 =  X[labels == 1,]
    # Define the variables
    beta = cp.Variable(2)
    beta0 = cp.Variable()

    # Define the constraints
    constraint1 = []
    constraint2 = []
    for i in range(class_1.shape[0]):
        constraint1 += [np.matmul(class_1[i,], beta) + beta0 <= -1]
    for j in range(class_2.shape[0]):
        constraint2 += [np.matmul(class_2[j,], beta) + beta0 >= 1]
    # Sum the constraints
    constraints = constraint1 + constraint2
    # Define the objective.
    obj = cp.Minimize((cp.norm(beta)**2)/2)
    # Add objective and constraint in the problem
    prob = cp.Problem(obj, constraints)
    # Solve the problem
    prob.solve()
    print("optimal var", beta.value, beta0.value)

    return beta.value, beta0.value


def svm_Slack(X, labels, C):
    # split the data in the two classes. Name them class_1 and class_2.
    ## Assign label 0 to class_1
    class_1 = X[labels == 0,]
    ## Assign label 1 to class_2
    class_2 =  X[labels == 1,]
    # Define the variables
    beta = cp.Variable(2)
    beta0 = cp.Variable()
    xi_1 = cp.Variable(class_1.shape[0])
    xi_2 = cp.Variable(class_2.shape[0])
    # Define the constraints
    constraint1 = []
    constraint2 = []
    for i in range(class_1.shape[0]):
        constraint1 += [np.matmul(class_1[i,], beta) + beta0 - xi_1[i] <= -1]
        constraint1 += [xi_1[i] >= 0]
    for j in range(class_2.shape[0]):
        constraint2 += [np.matmul(class_2[j,], beta) + beta0 + xi_2[j] >= 1]
        constraint2 += [xi_2[j] >= 0]

    # Sum the constraints
    constraints = constraint1 + constraint2
    # Define the objective. Hint: use cp.norm
    obj = cp.Minimize(C*(sum(xi_1) + sum(xi_2)) + (cp.norm(beta)**2)/2)
    # Add objective and constraint in the problem
    prob = cp.Problem(obj, constraints)
    # Solve the problem
    prob.solve()
    print("optimal var", beta.value, beta0.value)

    return beta.value, beta0.value


def plot_decision_boundary(X, y, theta0, theta):
    # X --> Inputs
    # theta --> parameters

    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = np.array([min(X[:, 0]), max(X[:, 0])])
    m = -theta[0] / theta[1]
    c = -theta0 / theta[1]
    x2 = m * x1 + c

    fig = plt.figure(figsize=(10, 8))
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "r^")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title('SVM')
    plt.plot(x1, x2, 'y-')


# results = svm_noSlack(X_noSlack, labels_noSlack)
# beta = results[0]
# beta0 = results[1]
# plot_decision_boundary(X_noSlack, labels_noSlack, beta0, beta)


results = svm_Slack(X_Slack, labels_Slack, C=1)
beta = results[0]
beta0 = results[1]
plot_decision_boundary(X_Slack, labels_Slack, beta0, beta)


# reference: https://github.com/learn-co-curriculum/dsc-building-an-svm-from-scratch-lab/blob/master/index.ipynb