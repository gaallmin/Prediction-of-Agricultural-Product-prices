import numpy as np


def nmae(
    y_hat: dict,
    y: dict
):

    m = len(y.keys())

    score = 0
    for item in y_hat.keys():
        n = y[item].shape[0]
        score += np.sum(np.abs(y_hat[item].flatten() - y[item].flatten())/y[item].flatten())/n

    score = score / m

    return score


def test(
    models: dict,
    x_val: dict,
    y_val: dict
):

    pred: dict = {}

    for item in models.keys():
        pred[item] = models[item].predict(x_val[item])
    print(f"{nmae(pred, y_val)}")


if __name__ == "__main__":

    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import VotingRegressor
    from data_loader import data_loader_v1

    x_train, x_val, y_train, y_val = data_loader_v1("./dataset/train/train.csv", output_size=1)
    for item in y_train.keys():
        y_train[item] = np.ravel(y_train[item])

    depth = 1

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

    test(models, x_val, y_val)
