import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

diabetes = datasets.load_diabetes()
X = diabetes['data']
y = diabetes['target'].reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)
# Fit model
model.fit(X_train, y_train)

lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)
y_test_pred= lasso_best.predict(X_test)

mse = (np.square(y_test_pred - y_test)).mean() # 8654.37
