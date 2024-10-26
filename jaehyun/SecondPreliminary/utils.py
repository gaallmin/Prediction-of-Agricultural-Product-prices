import numpy as np
from sklearn.model_selection import cross_val_score, KFold
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
    y: np.ndarray,
    y_hat: np.ndarray
):

    n = y.shape[0]
    score = np.sum(np.abs(y_hat.flatten() - y.flatten())/y.flatten())/n

    return score


def mean_raw_cv(
    models: dict,
    X: dict,
    y: dict,
    k: int = 4,
):

    score = [] 
    keys = X.keys()
    for item in keys:

        val_size = int(y[item]['common'].shape[0] / k)
        scores = []
        for i in range(k):
            x_common_split = np.split(X[item]['common'], [val_size*i, val_size*(i+1)])
            x_common_train = np.vstack((x_common_split[0], x_common_split[2]))
            x_common_val = x_common_split[1]

            y_common_split = np.split(y[item]['common'], [val_size*i, val_size*(i+1)])
            y_common_train = np.hstack((y_common_split[0], y_common_split[2]))
            y_common_val = y_common_split[1]

            x_mean_split = np.split(X[item]['mean'], [val_size*i, val_size*(i+1)])
            x_mean_train = np.vstack((x_mean_split[0], x_mean_split[2]))
            x_mean_val = x_mean_split[1]

            y_mean_split = np.split(y[item]['mean'], [val_size*i, val_size*(i+1)])
            y_mean_train = np.hstack((y_mean_split[0], y_mean_split[2]))
            y_mean_val = y_mean_split[1]

            models[item].fit(
                {
                    'common': x_common_train,
                    'mean': x_mean_train,
                },
                {
                    'common': y_common_train,
                    'mean': y_mean_train,
                },
            )

            pred = models[item].predict({
                'common': x_common_val,
                'mean': x_mean_val
            })

            scores.append(nmae_not_dict(
                (1 - models[item].mean_percentage)*y_common_val + (models[item].mean_percentage)*y_mean_val,
                pred,
            ))

        score.append(sum(scores)/len(scores))

    for i, item in enumerate(keys):
        print(f"{item}: {score[i]}")

    score = sum(score)/len(score)

    print(f"cv nmae: {score}")


def residual_raw_cv(
    models: dict,
    X: dict,
    y: dict,
    k: int = 4,
):

    score = [] 
    keys = X.keys()
    for item in keys:

        val_size = int(y[item]['common'].shape[0] / k)
        scores = []
        for i in range(k):
            x_common_split = np.split(X[item]['common'], [val_size*i, val_size*(i+1)])
            x_common_train = np.vstack((x_common_split[0], x_common_split[2]))
            x_common_val = x_common_split[1]

            y_common_split = np.split(y[item]['common'], [val_size*i, val_size*(i+1)])
            y_common_train = np.hstack((y_common_split[0], y_common_split[2]))
            y_common_val = y_common_split[1]

            x_residual_split = np.split(X[item]['residual'], [val_size*i, val_size*(i+1)])
            x_residual_train = np.vstack((x_residual_split[0], x_residual_split[2]))
            x_residual_val = x_residual_split[1]

            y_residual_split = np.split(y[item]['residual'], [val_size*i, val_size*(i+1)])
            y_residual_train = np.hstack((y_residual_split[0], y_residual_split[2]))
            y_residual_val = y_residual_split[1]

            models[item].fit(
                {
                    'common': x_common_train,
                    'residual': x_residual_train,
                },
                {
                    'common': y_common_train,
                    'residual': y_residual_train,
                },
            )

            pred = models[item].predict({
                'common': x_common_val,
                'residual': x_residual_val
            })

            scores.append(nmae_not_dict(
                y_common_val + y_residual_val,
                pred,
            ))

        score.append(sum(scores)/len(scores))

    for i, item in enumerate(keys):
        print(f"{item}: {score[i]}")

    score = sum(score)/len(score)

    print(f"cv nmae: {score}")


def raw_cv(
    models: dict,
    X: dict,
    y: dict,
    scaler: dict = None,
    k: int = 4,
):

    score = 0
    keys = X.keys()
    for item in keys:

        if scaler != None:
            y[item] = scaler[item].inverse_transform(y[item])

        r = np.arange(len(y[item]))
        val_size = int(y[item].shape[0] / k)
        scores = []
        for i in range(k):
            x_split = np.split(X[item], [val_size*i, val_size*(i+1)])
            x_train = np.vstack((x_split[0], x_split[2]))
            x_val = x_split[1]

            y_split = np.split(y[item], [val_size*i, val_size*(i+1)])
            y_train = np.hstack((y_split[0], y_split[2]))
            y_val = y_split[1]

            models[item].fit(x_train, y_train)

            pred = models[item].predict(x_val)

            if scaler != None:
                pred = scaler[item].inverse_transform(pred)

            scores.append(nmae_not_dict(y_val, pred))
        print(f"{item}: {sum(scores)/len(scores)}")
        score += sum(scores)/len(scores)

    score = score/len(keys)

    print(f"cv nmae: {score}")


def double_raw_cv(
    a_models: dict,  # Average price
    cya_models: dict,  # Common year average price
    a_X: dict,
    a_y: dict,
    cya_X: dict,
    cya_y: dict,
    k: int,
):

    score = 0
    keys = X.keys()
    for item in keys:

        val_size = int(y[item].shape[0] / k)
        scores = []
        for i in range(k):
            x_split = np.split(X[item], [val_size*i, val_size*(i+1)])
            x_train = np.vstack((x_split[0], x_split[2]))
            x_val = x_split[1]

            y_split = np.split(y[item], [val_size*i, val_size*(i+1)])
            y_train = np.hstack((y_split[0], y_split[2]))
            y_val = y_split[1]

            models[item].fit(x_train, y_train)
            scores.append(nmae_not_dict(y_val, models[item].predict(x_val)))
        score += sum(scores)/len(scores)
    score = score/len(keys)

    print(f"cv nmae: {score}")


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

    for idx, item in enumerate(models.keys()):
        print(f"{item}: {round(scores[idx], 5)}")

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


# 전 3개의 month를 받으면 다음 month를 출력
def next_month(
    prev_months: np.ndarray
):

    if np.unique(prev_months).shape != (1,):
        return prev_months[-1]
    else:
        return prev_months[0]


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
