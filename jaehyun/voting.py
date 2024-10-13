import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

from data_loader import data_loader_v1, data_loader
from submission import submit, submit_v2
from utils import test

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train/v1_averageFilling.csv",
    output_size=1,
    #new_features=['평년 평균가격(원)']
)
for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

cat_depth = 1
xgb_depth = 1

# 500: 0.09277262654108845
# 600: 0.09258870439987489
# 700: 0.09262657209138672
# 1000: 0.0930
cat_params = {
    'random_state': 2024,
    'n_estimators': 90,
    'learning_rate': 0.05,
    'depth': cat_depth,
    'l2_leaf_reg': 3,
}
xgb_params = {
    'n_estimators': 90,
    'random_state': 2024,
    "learning_rate": 0.05,
    'max_depth': xgb_depth,
}
models = {}
for item in x_train.keys():
    cat = CatBoostRegressor(**cat_params)
    xgb = XGBRegressor(**xgb_params)
    models[item] = VotingRegressor(
        estimators=[('cat', cat), ('xgb', xgb)]
    )
    models[item].fit(x_train[item], y_train[item])

test(models, x_val, y_val)

submit_v2(
    f"submission/voting_{cat_depth}_{xgb_depth}_600.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1
)
