import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor

from data_loader import data_loader
from submission import submit
from utils import test, raw_cv, cv

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train",
    output_size=1,
    train_percentage=1,
    process_method='log'
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

models = {}
for item in x_train.keys():

    models[item] = MLPRegressor(learning_rate_init=0.05, random_state=1, max_iter=5000, early_stopping=True)
    #models[item].fit(x_train[item], y_train[item])

cv(models, x_train, y_train)

'''
submit(
    f"submission/Lasso_multi_10_log.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
    process_method='ewma'
)
'''
