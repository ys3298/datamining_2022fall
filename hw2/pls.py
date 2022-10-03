import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
def pls(X_train, y_train, X_test, y_test, n_components):
    # X_train = StandardScaler().fit_transform(X_train) # n*p
    # X_test = StandardScaler().fit_transform(X_test)
    X_mean_train = np.average(X_train, axis = 0)
    X_var_train = np.var(X_train, axis = 0)
    X_train = (X_train-X_mean_train)/X_var_train

    X_mean_test = np.average(X_test, axis=0)
    X_var_test = np.var(X_test, axis=0)
    X_test = (X_test - X_mean_test) / X_var_test

    # 1. Get Z matrix
    X_temp = X_train # initialization n*p
    Z = np.zeros([X_train.shape[0], X_train.shape[1]])
    for m in range(X_train.shape[1]):

        ## Obtain the mth direction
        # phi = np.matmul(X_temp.T, y_train) # phi is a p*1 vector
        # Z[:,m] = np.matmul(X_temp, phi).reshape(-1,) # n*1

        for k in range(X_train.shape[1]):
            phi_j = np.matmul(X_temp[:,k].T,y_train)
            Z[:,m] = Z[:,m] + phi_j*X_temp[:,k]

        # Update xj(m)
        for j in range(X_train.shape[1]):
            X_temp[:,j] = X_temp[:,j] - (np.matmul(Z[:,m].T,X_temp[:,j])/np.matmul(Z[:,m].T,Z[:,m]))*Z[:,m]

    # 2. Regression
    reg = LinearRegression().fit(Z[:,0:n_components], y_train)
    Gamma = np.linalg.inv(np.matmul(np.matmul(np.matmul(X_temp.T, X_temp),X_temp.T),Z))
    Z_test = np.matmul(X_test, Gamma[:,0:n_components])


    y_test_pred = reg.predict(Z_test)
    # Mean Squared Error
    mse = (np.square(y_test_pred - y_test)).mean()

    return mse


## example data
diabetes = datasets.load_diabetes()
X = diabetes['data']
X = X[0:3,0:3]
y = diabetes['target'].reshape(-1,1)
y = y[0:3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
output = list()
for n in list(range(1, X.shape[1]+1)):
    print(n)

    mse = pls(X_train, y_train, X_test, y_test, n)
    mse = round(mse, 1)
    output.append(mse)

print(output)