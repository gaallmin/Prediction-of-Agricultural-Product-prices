import pandas as pd

def show_train():

    train = pd.read_csv("./dataset/train/train.csv")

    print(train.loc[
        (train['품목명'] == '감자') &
        (train['품종명'] == '감자 수미') &
        (train['등급'] == '상') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '건고추') &
        (train['품종명'] == '화건') &
        (train['등급'] == '상품') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '깐마늘(국산)') &
        (train['품종명'] == '깐마늘(국산)') &
        (train['등급'] == '상품') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '대파') &
        (train['품종명'] == '대파(일반)') &
        (train['등급'] == '상') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '무') &
        (train['품종명'] == '무') &
        (train['등급'] == '상') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '배') &
        (train['품종명'] == '신고') &
        (train['등급'] == '상품') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '배추') &
        (train['품종명'] == '배추') &
        (train['등급'] == '상') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '사과') &
        (train['품종명'] == '홍로') &
        (train['등급'] == '상품') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '사과') &
        (train['품종명'] == '후지') &
        (train['등급'] == '상품')
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '상추') &
        (train['품종명'] == '청') &
        (train['등급'] == '상품') &
        (train['평균가격(원)'] > 0)
    ])

    print(train.loc[
        (train['품목명'] == '양파') &
        (train['품종명'] == '양파') &
        (train['등급'] == '상') &
        (train['평균가격(원)'] > 0)
    ])
    '''
