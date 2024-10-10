from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from data_loader import data_loader_v1

x_train, x_val, y_train, y_val = data_loader_v1("./dataset/train/train.csv", output_size=1, train_percentage=1)
for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

depth = 2

cat_params = {
    'random_state': 2024,
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'depth': depth,
    'l2_leaf_reg': 3,
}
xgb_params = {
    'n_estimators': 1000,
    'random_state': 2024,
    "learning_rate": 0.05,
    'max_depth': depth,
}
models = {}
for item in x_train.keys():
    cat = CatBoostRegressor(**cat_params)
    xgb = XGBRegressor(**xgb_params)
    models[item] = VotingRegressor(
        estimators=[('cat', cat), ('xgb', xgb)]
    )
    models[item].fit(x_train[item], y_train[item])

submit(
    f"submission/voting_{depth}.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1
)
