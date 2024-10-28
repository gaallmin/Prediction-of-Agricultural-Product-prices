from sys import path
path.append('../')

from os import listdir
from os.path import join, isfile
from itertools import product

import pandas as pd
import numpy as np

from data_loader_meta import preprocess_month


PRODUCT_CASE = [
    "배추",
    "무",
    "양파",
    "감자",
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

def join_weather(
    data_dict,
    path: str = "../dataset/train/meta/TRAIN_기상_2018-2022.csv",
    join: dict = {
        # 리스트는 '품종', '지역', '피쳐' 순
        '배추': ['가을', 'B', '순 최저상대습도'],
        '건고추': ['-', 'E', '순 평균풍속'],
        '무': ['가을', 'B', '순 강수량'],
        '양파': ['중만생종', 'K', '순 최고기온'],
    },
):

    data = pd.read_csv(path)

    for item in join.keys():

        weather_feature = join[item][2:]

        new_data = {}
        yyyymmsoon_list = data_dict[item]['YYYYMMSOON'].to_list()
        product_list = []
        feature_list = []

        for yyyymmsoon in yyyymmsoon_list:
            product_list.append(item)
            feature_list.append(data[
                (data['YYYYMMSOON'] == yyyymmsoon) &
                (data['주산지 품목명'] == PRODUCT_CASE[CASE.index(item)]) &
                (data['주산지 품종명'] == join[item][0]) &
                (data['지역 이름'] == join[item][1])
            ][weather_feature].iloc[0].to_numpy())

        feature_list = np.array(feature_list)

        new_data['품목명'] = product_list
        new_data['YYYYMMSOON'] = yyyymmsoon_list

        for idx, feature in enumerate(weather_feature):
            new_data[feature] = feature_list[:, idx]

        new_data = pd.DataFrame(new_data)

        data_dict[item] = data_dict[item].join(new_data[['YYYYMMSOON'] + weather_feature].set_index('YYYYMMSOON'), on='YYYYMMSOON')

    return data_dict

def submit(
    submit_name: str,
    test_folder: str,
    sample_submission_file: str,
    models: list,
    input_size: int = 9,
    process_method: str = 'ewm',
    feature: list = ['평균가격(원)'],
    is_month: bool = False,
    weather_path: str = "../dataset/test/meta/",
    weather: dict = {
        # 리스트는 '품종', '지역', '피쳐' 순
        '배추': ['가을', 'B', '순 최저상대습도'],
        '건고추': ['-', 'E', '순 평균풍속'],
        '무': ['가을', 'B', '순 강수량'],
        '양파': ['중만생종', 'K', '순 최고기온'],
    },
):

    # 3으로 고정
    OUTPUT_SIZE = 3

    pred = {}

    file_numbers = [i for i in range(52)]

    test_dict = {}
    for item in CASE:
        test_dict[item] = []

    for file_number in file_numbers: 

        tmp_dict = {}

        for item in CASE[:5]:
            test_file = f"TEST_{str(file_number).zfill(2)}_1.csv"
            test = pd.read_csv(join(test_folder, test_file))
            condition = test['품목(품종)명'] == item
            tmp_dict[item] = test.loc[condition]

        for item in CASE[5:]:
            test_file = f"TEST_{str(file_number).zfill(2)}_2.csv"
            test = pd.read_csv(join(test_folder, test_file))
            condition = test['품목명'] == item
            tmp_dict[item] = test.loc[condition]

        for item in CASE:
            if is_month:
                tmp_dict[item] = preprocess_month(tmp_dict[item])

        tmp_dict = join_weather(
            tmp_dict,
            path=join(test_folder, f'meta/TEST_기상_{str(file_number).zfill(2)}.csv'),
            join=weather,
        )

        for item in CASE:

            feature_list = feature

            if item in weather.keys():
                weather_feature = weather[item][2:]
                feature_list = feature_list + weather_feature
            if is_month:
                feature_list = feature_list + ['Month']

            tmp_dict[item] = tmp_dict[item][feature_list]
            
            if process_method == 'ewm':
                tmp_dict[item]['평균가격(원)'] = tmp_dict[item]['평균가격(원)'].ewm(alpha=0.4).mean()
            elif process_method == 'ewma':
                tmp_dict[item]['평균가격(원)'] = tmp_dict[item]['평균가격(원)'].ewm(span=4, adjust=False).mean()
            elif process_method == 'sma':
                tmp_dict[item]['평균가격(원)'] = tmp_dict[item]['평균가격(원)'].rolling(window=3, min_periods=1).mean()
            elif process_method == 'log':
                pass

            test_dict[item].append(tmp_dict[item].to_numpy())

    for item in CASE:
        test_dict[item] = np.array(test_dict[item])

        pred[item] = models[item].predict(test_dict[item])
        pred[item] = pred[item].flatten()

    submission = pd.read_csv(sample_submission_file)
    for item in CASE:
        submission[item] = pred[item]

    submission.to_csv(submit_name, index=False)
