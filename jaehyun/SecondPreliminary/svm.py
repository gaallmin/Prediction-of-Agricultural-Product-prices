import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import data_loader
from submission import submit
from utils import test, raw_cv, cv

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train",
    output_size=1,
    train_percentage=1,
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

models = {}
for item in x_train.keys():

    #lasso = [Lasso(alpha=1, tol=1e-7, selection='random')]*10
    #models[item] = VotingRegressor(
    #    estimators=[
    #        (f'lasso_{i}', lasso[i]) for i in range(10)
    #    ],
    #)

    models[item] = make_pipeline(StandardScaler(), SVR(C=20, epsilon=0.3))
    #models[item].fit(x_train[item], y_train[item])

raw_cv(models, x_train, y_train)

'''
submit(
    f"submission/Lasso_multi_10_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
)
'''
