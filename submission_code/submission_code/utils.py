import numpy as np


def nmae(
    y: np.ndarray,
    y_hat: np.ndarray
):

    n = y.shape[0]
    score = np.sum(np.abs(y_hat.flatten() - y.flatten()) / y.flatten()) / n

    return score


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

        if scaler is not None:
            y[item] = scaler[item].inverse_transform(y[item])

        val_size = int(y[item].shape[0] / k)
        scores = []
        for i in range(k):
            x_split = np.split(X[item], [val_size * i, val_size * (i + 1)])
            x_train = np.vstack((x_split[0], x_split[2]))
            x_val = x_split[1]

            y_split = np.split(y[item], [val_size * i, val_size * (i + 1)])
            y_train = np.hstack((y_split[0], y_split[2]))
            y_val = y_split[1]

            models[item].fit(x_train, y_train)

            pred = models[item].predict(x_val)

            if scaler is not None:
                pred = scaler[item].inverse_transform(pred)

            scores.append(nmae(y_val, pred))
        print(f"{item}: {sum(scores)/len(scores)}")
        score += sum(scores) / len(scores)

    score = score / len(keys)

    print(f"cv nmae: {score}")
