import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.ensemble import VotingRegressor, StackingRegressor

from data_loader import data_loader
from submission import submit
from utils import test, raw_cv, cv

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train",
    output_size=1,
    train_percentage=1,
    process_method='ewma'
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

models = {}
for item in x_train.keys():

    lasso = [Lasso(alpha=1, tol=1e-7, selection='random')]*10
    models[item] = StackingRegressor(
        estimators=[
            (f'lasso_{i}', lasso[i]) for i in range(10)
        ],
        final_estimator = KernelRidge(alpha=0.5)
    )
    models[item].fit(x_train[item], y_train[item])

#cv(models, x_train, y_train)

submit(
    f"submission/stakcing_lasso_10_kernel_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
    process_method='ewma'
)
