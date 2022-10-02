import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Three steps of PCA:
# 1. Scale the data — we don’t want some feature to be voted as “more important” due to scale differences.
# 2. Calculate covariance matrix — square matrix giving the covariances between each pair of elements of a random vector
# 3. Eigendecomposition

def pca(X, y, n_components):
    # step1
    X_scaled = StandardScaler().fit_transform(X)
    predictors = X_scaled.T
    # step2
    cov_matrix = np.cov(predictors)
    # step3
    values, vectors = np.linalg.eig(cov_matrix)

    # percentage of explained variance
    # explained_variances = []
    # for i in range(len(values)):
    #     explained_variances.append(values[i] / np.sum(values))

    # regression
    reg = LinearRegression().fit(vectors, y)
    # return
