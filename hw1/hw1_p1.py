import numpy as np
from sklearn.linear_model import LinearRegression
n = 10000
x0 = np.repeat(1,n)
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.3, n)
x3 = np.random.normal(0, 1, n)
# print(x1.shape)
x = np.array([x1, x2, x3]).T

error = np.random.normal(0, 1, n)
beta1 = 1.5
beta2 = 2.5
beta3 = 3.5
y = beta1*x1 + beta2*x2 + beta3*x3 + error

z0 = x0
gamma01 = np.inner(z0,x1)/np.inner(z0,z0)
z1 = x1 - gamma01*z0

gamma02 = np.inner(z0,x2)/np.inner(z0,z0)
gamma12 = np.inner(z1,x2)/np.inner(z1,z1)
z2 = x2 - gamma02*z0 - gamma12*z1

gamma03 = np.inner(z0,x3)/np.inner(z0,z0)
gamma13 = np.inner(z1,x3)/np.inner(z1,z1)
gamma23 = np.inner(z2,x3)/np.inner(z2,z2)
z3 = x3 - gamma03*z0 - gamma13*z1 - gamma23*z2

model = LinearRegression()
model.fit(z1.reshape(-1, 1), y)
print(f"coefficients: {model.coef_}")
model.fit(z2.reshape(-1, 1), y)
print(f"coefficients: {model.coef_}")
model.fit(z3.reshape(-1, 1), y)
print(f"coefficients: {model.coef_}")

model.fit(x,y)
print(f"coefficients: {model.coef_}")

# The last coefficient of regression is numerically the same with the results of gram-smith process.
# While the others are pretty close, but not exatly the same.

