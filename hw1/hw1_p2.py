import numpy as np
import pandas as pd
from sklearn import preprocessing
# from matplotlib import pyplot as plt

## import and clean the data;
outcome = pd.read_table('gene_expression_sample/GEUVADIS_normalized_expression_chr20', sep ='\t')
choose = 0
start = outcome.loc[choose,'start']
end= outcome.loc[choose,'end']
outcome1 = outcome.loc[:, (outcome.columns.str.startswith('HG'))|(outcome.columns.str.startswith('NA'))]
y = outcome1.loc[choose,:].to_frame() # 358*1

raw = pd.read_table('gene_expression_sample/GEUVADIS_chr20_processed.traw', sep ='\t')
raw = raw[raw.POS.isin(range(start-500000, end+500000))]
df1 = raw.loc[:, (raw.columns =='SNP') | (raw.columns.str.startswith('HG')) | (raw.columns.str.startswith('NA'))]
df1 = df1.rename(columns=lambda s: s.split('_')[0])
df2 = df1.T # the covariate matrix, each row is a subject, each column is a variable
df2 = df2.rename(columns=df2.iloc[0]).drop(df2.index[0])

data = y.merge(df2, left_index=True, right_index=True)
data.rename(columns={0: "y"}, inplace=True)

y = data.iloc[:,0].to_numpy().reshape(-1,1)
X = data.iloc[:,1:data.shape[1]].to_numpy()
# y = data.iloc[1:101:,0].to_numpy().reshape(-1,1)
# X = data.iloc[1:101,1:101].to_numpy()

### function for algorithm
def close_form(rho,gamma,lambda_i):
    if rho > lambda_i:
        return (rho - lambda_i)/gamma
    elif rho < -lambda_i:
        return (rho + lambda_i)/gamma
    else:
        return 0

def coordinate_descent_lasso(X, y, lambda_inp = 0.001, epsilon=0.00001):
    # X = preprocessing.normalize(X.T).T # normalization
    n_sub = X.shape[0]
    n_pre = X.shape[1]
    beta = np.ones((n_pre,1))
    beta_old = np.copy(beta)
    ite = 1
    while ite == 1 or np.sum(np.square(beta_old - beta)) > epsilon:
        ite += 1
        beta_old = np.copy(beta)
        if ite % 100 == 1:
            # print(np.sum(np.square(beta_old - beta)))
            print(ite)
        for j in range(n_pre):
            xj_col_vec = X[:,j].reshape(-1,1)
            rho = (((xj_col_vec.T) @ (y - X @ beta + beta[j]*xj_col_vec)))/n_sub
            gamma = (xj_col_vec.T) @ (xj_col_vec)/n_sub

            # rho = (((xj_col_vec.T) @ (y - X @ beta + beta[j]*xj_col_vec))) / 1
            # gamma = 1
            beta[j] = close_form(rho, gamma, lambda_inp)

    return beta


# # lambda_inp = np.arange(0, 1, 1).tolist()
# lambda_inp = np.logspace(0,4,300)/10 #Range of lambda values
# beta_list = list()
# for lambda_now in lambda_inp:
#     beta = coordinate_descent_lasso(X, y, lambda_now)
#     beta_list.append(beta)
#
# test = np.reshape(beta_list, (len(lambda_inp), X.shape[1]))
# plt.plot(lambda_inp, test)
# plt.xscale('log')
# plt.xlabel('Log($\\lambda$)')
# plt.legend()

X = preprocessing.normalize(X.T).T
lambda_inp = 0.1
beta = coordinate_descent_lasso(X, y, lambda_inp, epsilon=0.00000001)
# np.savetxt("algorithm.csv", beta, delimiter=",")

