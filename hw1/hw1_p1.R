n=100000
set.seed(1000)
x0 = rep(1,n)
x1 = rnorm(n)
x2 = rbinom(n,1,0.3)
x3 = rnorm(n)
beta1 = 1.5
beta2 = 2.5
beta3 = 3.5
y = beta1*x1 + beta2*x2 + beta3*x3 +  rnorm(n)

z0 = x0
gamma01 = sum(z0*x1)/sum(z0*z0)
z1 = x1 - gamma01*z0

gamma02 = sum(z0*x2)/sum(z0*z0)
gamma12 = sum(z1*x2)/sum(z1*z1)
z2 = x2 - gamma02*z0 - gamma12*z1

gamma03 = sum(z0*x3)/sum(z0*z0)
gamma13 = sum(z1*x3)/sum(z1*z1)
gamma23 = sum(z2*x3)/sum(z2*z2)
z3 = x3 - gamma03*z0 - gamma13*z1 - gamma23*z2
lm(y~x1+x2+x3)
lm(y~z1)
lm(y~z2)
lm(y~z3)

# The last coefficient of regression is numerically the same with the results of gram-smith process. While the others are pretty close, but not exatly the same.
