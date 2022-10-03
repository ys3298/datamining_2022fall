import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

def pca(X_train, y_train, X_test, y_test, n_components):
    # X_train = StandardScaler().fit_transform(X_train)
    # X_test = StandardScaler().fit_transform(X_test)

    X_mean_train = np.average(X_train, axis=0)
    X_var_train = np.var(X_train, axis=0)
    X_train = (X_train - X_mean_train) / X_var_train

    X_mean_test = np.average(X_test, axis=0)
    X_var_test = np.var(X_test, axis=0)
    X_test = (X_test - X_mean_test) / X_var_test

    u, s, vh = np.linalg.svd(X_train)
    Z = np.matmul(X_train, vh)

    # regression on the training set
    reg = LinearRegression().fit(Z[:,0:n_components], y_train)
    # predict on the test set
    Z_test = np.matmul(X_test, vh[:,0:n_components])
    y_test_pred = reg.predict(Z_test)
    # Mean Squared Error
    mse = (np.square(y_test_pred - y_test)).mean()

    return mse


## example data
diabetes = datasets.load_diabetes()
X = diabetes['data']
y = diabetes['target'].reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

output = list()
for n in list(range(1, X.shape[1]+1)):
    print(n)
    mse = pca(X_train, y_train, X_test, y_test, n)
    mse = round(mse,1)
    output.append(mse)

print(output)
# [4204.9, 4024.6, 3976.3, 4119.7, 4044.6, 3783.8, 2859.4, 2820.1, 2821.8, 2800.2]