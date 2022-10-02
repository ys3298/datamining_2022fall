import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
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

X = preprocessing.normalize(X.T).T
model = linear_model.Lasso(alpha=0.1, fit_intercept=False, max_iter=10000000)
model.fit(X, y)
beta = model.coef_
# np.savetxt("sklearn.csv", model.coef_, delimiter=",")


# test2 = data.columns.values.tolist()
# df = pd.DataFrame(test2)
# df.to_excel('output_new.xlsx', header=False, index=False)

# Toy example:
# X=np.array([[-1,-4],[0,0],[1,16]])
# y=np.array([[-2.2],[0],[3.8]]).reshape(3,)
# X = preprocessing.normalize(X.T).T
# model = linear_model.Lasso(alpha=0.1, fit_intercept=False, max_iter=10000000)
# model.fit(X, y)
# beta = model.coef_