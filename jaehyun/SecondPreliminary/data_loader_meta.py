from os import path
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PRODUCT_CASE = [
    "배추",
    "무",
    "양파",
    "감자 수미",
    "대파",
    "건고추",
    "깐마늘",
    "상추",
    "사과",
    "배",
]

CASE = [
    "배추",
    "무",
    "양파",
    "감자 수미",
    "대파(일반)",
    "건고추",
    "깐마늘(국산)",
    "상추",
    "사과",
    "배",
]


def preprocess_month(
    data: pd.DataFrame,
):

    data['Month'] = data['YYYYMMSOON'].apply(lambda s: int(s[4:6]))

    return data

def join_weather(
    data_dict,
    path: str = "./dataset/train/meta/TRAIN_기상_2018-2022.csv",
    join: dict = {
        # 리스트는 '품종', '지역', '피쳐' 순
        # 결측은 NULL로 처리
        '배추': ['가을', 'A', '순 강수량'],
        '무': ['월동', 'D', '순 평균풍속'],
        '양파': ['중만생종', 'K', '순 평균기온'],
    },
):

    years = ['2018', '2019', '2020', '2021', '2022']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    data = pd.read_csv(path)

    for item in join.keys():

        soon = ['상순', '중순', '하순']

        new_data = {}
        product_list = []
        yyyymmsoon_list = []
        feature_list = []

        for year, month, soon in product(years, months, soon):
            product_list.append(item)
            yyyymmsoon_list.append(f'{year}{month}{soon}')
            feature_list.append(data[
                (data['YYYYMMSOON'] == f'{year}{month}{soon}') &
                (data['주산지 품목명'] == PRODUCT_CASE[CASE.index(item)]) &
                (data['주산지 품종명'] == join[item][0]) &
                (data['지역 이름'] == join[item][1])
            ][join[item][2]].iloc[0])

        new_data['품목명'] = product_list
        new_data['YYYYMMSOON'] = yyyymmsoon_list
        new_data[join[item][2]] = feature_list
        new_data = pd.DataFrame(new_data)

        data_dict[item] = data_dict[item].join(new_data[join[item][2]])

    return data_dict


def data_loader_meta(
    train_path: str = "./dataset/train",
    input_size: int = 9,
    train_percentage: float = 0.7,
    is_month: bool = False,
    weather: dict = {
        # 리스트는 '품종', '지역', '피쳐' 순
        '배추': ['가을', 'A', '순 강수량'],
        '무': ['월동', 'D', '순 평균풍속'],
        '양파': ['중만생종', 'K', '순 평균기온'],
    },
    process_method: str = 'ewm'  # 'ewm', sma', 'ewma', 'log'
):

    OUTPUT_SIZE = 3
    RANDOM_STATE = 9999

    data_1 = pd.read_csv(path.join(train_path, "train_1.csv"))
    data_2 = pd.read_csv(path.join(train_path, "train_2.csv"))

    if is_month:  # Month as one-hot
        data_1 = preprocess_month(data_1)
        data_2 = preprocess_month(data_2)

    x_train = {
        "배추": [],
        "무": [],
        "양파": [],
        "감자 수미": [],
        "대파(일반)": [],
        "건고추": [],
        "깐마늘(국산)": [],
        "상추": [],
        "사과": [],
        "배": [],
    }
    y_train = deepcopy(x_train)
    x_val = deepcopy(x_train)
    y_val = deepcopy(x_train)

    len_data = {}
    data_dict = {}
    for item in CASE[:5]:
        condition = data_1['품목(품종)명'] == item
        data_dict[item] = data_1.loc[condition]
        len_data[item] = data_dict[item].shape[0]

    for item in CASE[5:]:
        condition = data_2['품목명'] == item
        data_dict[item] = data_2.loc[condition]
        len_data[item] = data_dict[item].shape[0]

    data_dict = join_weather(data_dict, join=weather)

    input = {}
    for item in CASE:
        input[item] = ['평균가격(원)']
        if item in weather.keys():
            input[item] = input[item] + [weather[item][2]]
        if is_month:
            input[item] = input[item] + ['Month']

    output = ['평균가격(원)']
    for item in CASE:
        for idx in range(len_data[item] - input_size - OUTPUT_SIZE):
            x = data_dict[item].iloc[idx: idx + input_size][input[item]]
            y = data_dict[item].iloc[idx + input_size: idx + input_size + OUTPUT_SIZE][output]

            if process_method == 'ewm':
                x['평균가격(원)'] = x['평균가격(원)'].ewm(alpha=0.4).mean()
            elif process_method == 'ewma':
                x['평균가격(원)'] = x['평균가격(원)'].ewm(span=4, adjust=False).mean()
            elif process_method == 'sma':
                x['평균가격(원)'] = x['평균가격(원)'].rolling(window=3, min_periods=1).mean()
            elif process_method == 'log':
                pass

            x = x.to_numpy()
            y = y.to_numpy().flatten()

            x_train[item].append(x)
            y_train[item].append(y)

        x_train[item] = np.array(x_train[item])
        y_train[item] = np.array(y_train[item])

        if train_percentage < 1:
            x_train[item], x_val[item], y_train[item], y_val[item] = train_test_split(
                x_train[item],
                y_train[item],
                test_size=1 - train_percentage,
                random_state=RANDOM_STATE
            )
        else:
            x_val[item] = None
            y_val[item] = None

    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    x_train, x_val, y_train, y_val = data_loader_meta(process_method='ewma')

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense

    input_sequence_length = 9  # 평균가격과 강수량 각 9개
    output_sequence_length = 3  # 예측할 미래의 평균가격 3개
    num_features = 2  # 평균가격, 강수량 두 가지 입력 피처

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(input_sequence_length, num_features)),
        Dense(32, activation='relu'),
        Dense(output_sequence_length)
    ])

    cb_early_stopping = EarlyStopping(
        monitor='loss',
        mode='min',
        patience=10,
    )

    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-10,
        verbose=1,
        min_delta=1e-5
    )

    model.compile(optimizer='adam', loss='mse')

    model.fit(
        x_train['배추'],
        y_train['배추'],
        epochs=5000,
        batch_size=64,
        validation_data=(x_val['배추'], y_val['배추']),
        callbacks=[cb_early_stopping, rlr],
    )
