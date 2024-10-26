from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np

from dummy import LastPredictor
from data_loader import data_loader
from data_loader_test import data_loader_comb, data_loader_common
from submission import submit, submit_comb
from utils import test, raw_cv, cv, residual_raw_cv


# Custom model combining Lasso and DecisionTree
class LassoTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tree_depth=5):
        self.tree_depth = tree_depth

        self.first = LastPredictor()

        lasso = [Lasso(alpha=1, tol=1e-7, selection='random')]*10
        self.second = VotingRegressor(
            estimators=[
                (f'second_{i}', lasso[i]) for i in range(10)
            ],
        )


    def fit(self, X, y):
        # Step 1: Fit the Lasso model
        self.first.fit(X, y)
        # Step 2: Calculate residuals
        residuals = y - self.first.predict(X)
        # Step 3: Fit the tree on residuals
        self.second.fit(X, residuals)
        return self

    def predict(self, X):
        # Predict using Lasso and add the residuals predicted by the tree
        first_preds = self.first.predict(X)
        second_preds = self.second.predict(X)
        return first_preds + second_preds


# Custom model combining Lasso and DecisionTree
class CombinedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):

        self.common_model = LassoTreeRegressor()

    def fit(self, X, y):

        self.common_model.fit(X['common'], y['common'])

        return self

    def predict(self, X):
        # Predict using Lasso and add the residuals predicted by the tree
        common_preds = self.common_model.predict(X['common'])

        return common_preds + np.mean(X['residual'], axis=1)

'''
배추: 0.16806459753454314
무: 0.1285070501099372
양파: 0.10006963261347707
감자 수미: 0.1060208266860644
대파(일반): 0.1404766736619056
건고추: 0.07600089606196334
깐마늘(국산): 0.019255475654193395
상추: 0.10218760081335508
사과: 0.08557062212591401
배: 0.0326263118341509
cv nmae: 0.09587796870955043
'''

x_train, _, y_train, _, x_common_train, _, y_common_train, _ = data_loader_common(
    train_path="./dataset/train",
    #x_comb=x_comb,
    input_size=3,
    output_size=1,
    train_percentage=1, process_method='ewma'
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])
    y_common_train[item] = np.ravel(y_common_train[item])

X = {}
y = {}
for item in y_train.keys():

    X[item] = {
        'common': x_common_train[item],
        'residual': x_train[item] - x_common_train[item],
    }
    y[item] = {
        'common': y_common_train[item],
        'residual': y_train[item] - y_common_train[item],
    }


models = {}
for item in x_train.keys():

    models[item] = CombinedRegressor()
    #models[item].fit(X[item], y[item])

residual_raw_cv(models, X, y)

'''
submit(
    f"submission/input_3_residual_combined_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
    input_size=3,
    process_method='ewma'
)
'''
