import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

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


def nmae_not_dict(
    y: dict,
    y_hat: dict
):

    n = y.shape[0]
    score = np.sum(np.abs(y_hat.flatten() - y.flatten())/y.flatten())/n

    return score


def cv(
    models: dict,
    X: dict,
    y: dict
):

    nmae_score = make_scorer(nmae_not_dict)

    scores: list = []
    for item in models.keys():
        score = cross_val_score(
            models[item],
            X[item],
            y[item],
            cv=4,
            scoring=nmae_score
        )
        scores.append(np.mean(score))
    mean_score = sum(scores)/len(scores)

    print(f"cv nmae: {mean_score}")


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
