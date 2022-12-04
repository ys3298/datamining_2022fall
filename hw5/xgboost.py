import numpy as np
# simulate data. n=100, p=10, multivariate gaussiasn distribution
mean = np.zeros(10)
cov = np.identity(10)
X = np.random.multivariate_normal(mean, cov, 120)
sum_of_squares = np.sum(np.square(X), axis=-1)

# the median of chi-square distribution approximately equal to df - 0.7, where we set df = 10
df = 10
median = df - 0.7

# generate y by comparing the sum of square with median of chi-square distribution
y = 1*(sum_of_squares >= median)

X_train = X[0:100,]
X_test = X[100:,]
y_train = y[0:100]
y_test = y[100:]


# # helper functions
# def similarity_score(pred_prob, true_y, Lambda = 1):
#     residual = true_y - pred_prob
#     residual_square = np.square(np.sum(residual))
#     cover = np.sum(pred_prob * (1-pred_prob))
#     score = residual_square / (cover + Lambda)
#     return scorea
#
#
# def output_value(ind, pred_prob, X, true_y, Lambda = 1):
#     residual = true_y - pred_prob
#     residual_sum = np.sum(residual)
#     cover = np.sum(pred_prob * (1 - pred_prob))
#     output = residual_sum  / (cover + Lambda)
#     return output
#
#
# def Gain(left_ind, right_ind)


class Node:
    def __init__(self, x, gradient, hessian, idxs, subsample_cols=1, depth=1,lambda_=1, gamma=0, eps=0.1):

        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs
        self.depth = depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols * self.col_count)]

        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])

        self.score = float('-inf')
        self.find_varsplit()

    def compute_gamma(self, gradient, hessian):
        return (-np.sum(gradient) / (np.sum(hessian) + self.lambda_))

    def find_varsplit(self):
        for c in self.column_subsample: self.find_greedy_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[lhs],
                        min_leaf=self.min_leaf, depth=self.depth - 1, lambda_=self.lambda_, gamma=self.gamma,
                        min_child_weight=self.min_child_weight, eps=self.eps, subsample_cols=self.subsample_cols)
        self.rhs = Node(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[rhs],
                        min_leaf=self.min_leaf, depth=self.depth - 1, lambda_=self.lambda_, gamma=self.gamma,
                        min_child_weight=self.min_child_weight, eps=self.eps, subsample_cols=self.subsample_cols)

    def find_greedy_split(self, var_idx):
        x = self.x.values[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]

            lhs_indices = np.nonzero(x <= x[r])[0]
            rhs_indices = np.nonzero(x > x[r])[0]
            if (rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf
                    or self.hessian[lhs_indices].sum() < self.min_child_weight
                    or self.hessian[rhs_indices].sum() < self.min_child_weight): continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def gain(self, lhs, rhs):
        gradient = self.gradient[self.idxs]
        hessian = self.hessian[self.idxs]

        lhs_gradient = gradient[lhs].sum()
        lhs_hessian = hessian[lhs].sum()

        rhs_gradient = gradient[rhs].sum()
        rhs_hessian = hessian[rhs].sum()

        gain = 0.5 * ((lhs_gradient ** 2 / (lhs_hessian + self.lambda_)) + (
                    rhs_gradient ** 2 / (rhs_hessian + self.lambda_)) - ((lhs_gradient + rhs_gradient) ** 2 / (
                    lhs_hessian + rhs_hessian + self.lambda_))) - self.gamma
        return (gain)

    @property
    def split_col(self):
        '''
        splits a column
        '''
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        '''
        checks if node is a leaf
        '''
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return (self.val)

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)


class XGBoostClassifier:
    '''
    Full application of the XGBoost algorithm as described in "XGBoost: A Scalable Tree Boosting System" for
    Binary Classification.
    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score).
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to
    (1 / sketch_eps) number of bins
    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    '''

    def __init__(self):
        self.estimators = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # first order gradient logLoss
    def grad(self, preds, labels):
        preds = self.sigmoid(preds)
        return (preds - labels)

    # second order gradient logLoss
    def hess(self, preds, labels):
        preds = self.sigmoid(preds)
        return (preds * (1 - preds))

    @staticmethod
    def log_odds(column):
        binary_yes = np.count_nonzero(column == 1)
        binary_no = np.count_nonzero(column == 0)
        return (np.log(binary_yes / binary_no))

    def fit(self, X, y, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5, learning_rate=0.4,
            boosting_rounds=5, lambda_=1.5, gamma=1, eps=0.1):
        self.X, self.y = X, y.values
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma

        self.base_pred = np.full((X.shape[0], 1), 1).flatten().astype('float64')

        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth=self.depth, min_leaf=self.min_leaf,
                                              lambda_=self.lambda_, gamma=self.gamma, eps=self.eps,
                                              min_child_weight=self.min_child_weight,
                                              subsample_cols=self.subsample_cols)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)

    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])

        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)

        return (self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred))

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)

        predicted_probas = self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return (preds)