import pandas as pd

def add_2017():
    df1 = pd.read_csv("./dataset/train/train_1.csv")
    df2 = pd.read_csv("./dataset/train/train_2.csv")

    # 2018년에서 전년도 데이터를 가져와서 2017년을 추가
    # 배추 사용 가능
    # 무는 사용 불가능
    # 양파 사용 가능
    # 감자 수미 사용 가능
    # 대파(일반) 사용 가능
    # 건고추 사용 가능
    # 깐마늘 사용 불가능
    # 상추 사용 가능
    # 사과 사용 가능 (약간의 결측치)
    # 배 사용 가능

    not_case = {
        "무": df1[df1['품목(품종)명'] == "무"],
        "깐마늘(국산)": df2[df2['품목명'] == "깐마늘(국산)"],
        "사과": df2[df2['품목명'] == "사과"],  # 사과는 결측이 있어서 제거
    }

    case = {
        "배추": df1[df1['품목(품종)명'] == "배추"],
        "양파": df1[df1['품목(품종)명'] == "양파"],
        "감자 수미": df1[df1['품목(품종)명'] == "감자 수미"],
        "대파(일반)": df1[df1['품목(품종)명'] == "대파(일반)"],
        "건고추": df2[df2['품목명'] == "건고추"],
        "상추": df2[df2['품목명'] == "상추"],
        "배": df2[df2['품목명'] == "배"],
    }

    YEAR = '2017'
    MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    yyyymmsoon = []
    for month in MONTHS:
        for soon in ['상순', '중순', '하순']:
            yyyymmsoon.append(YEAR + month + soon)

    for item, item_df in case.items():
        target = item_df.head(12*3).copy()
        target['YYYYMMSOON'] = yyyymmsoon
        target['평균가격(원)'] = target['전년 평균가격(원) PreVious YeaR']  # 18년의 전년 평균가격을 17년의 평균가격으로
        target['전순 평균가격(원) PreVious SOON'] = 0.0
        target['전달 평균가격(원) PreVious MMonth'] = 0.0
        target['전년 평균가격(원) PreVious YeaR'] = 0.0
        target['평년 평균가격(원) Common Year SOON'] = 0.0

        case[item] = pd.concat([target, item_df], ignore_index=True)

    new_df1 = pd.concat([case['배추'], not_case['무'], case['양파'], case['감자 수미'], case['대파(일반)']], ignore_index=True)
    new_df2 = pd.concat([case['건고추'], not_case['깐마늘(국산)'], case['상추'], not_case['사과'], case['배']], ignore_index=True)

    new_df1.to_csv("./dataset/train/train_1_v2.csv")
    new_df2.to_csv("./dataset/train/train_2_v2.csv")


def remove():
    df1 = pd.read_csv("./dataset/train/train_1.csv")
    df2 = pd.read_csv("./dataset/train/train_2.csv")

    # 2018년, 2019년 데이터를 제거해보자

    case = {
        "배추": df1[df1['품목(품종)명'] == "배추"],
        "무": df1[df1['품목(품종)명'] == "무"],
        "양파": df1[df1['품목(품종)명'] == "양파"],
        "감자 수미": df1[df1['품목(품종)명'] == "감자 수미"],
        "대파(일반)": df1[df1['품목(품종)명'] == "대파(일반)"],
        "건고추": df2[df2['품목명'] == "건고추"],
        "깐마늘(국산)": df2[df2['품목명'] == "깐마늘(국산)"],
        "상추": df2[df2['품목명'] == "상추"],
        "사과": df2[df2['품목명'] == "사과"],
        "배": df2[df2['품목명'] == "배"],
    }

    # 줄이기 전
    # 배추: 0.16005
    # 무: 0.12851
    # 양파: 0.09987
    # 감자 수미: 0.10282
    # 대파(일반): 0.14042
    # 건고추: 0.02608
    # 깐마늘(국산): 0.01924
    # 상추: 0.09778
    # 사과: 0.0324
    # 배: 0.03331

    # 줄인 후
    # 배추: 0.15436
    # 무: 0.12682
    # 양파: 0.10844
    # 감자 수미: 0.09086
    # 대파(일반): 0.16043
    # 건고추: 0.02548
    # 깐마늘(국산): 0.01676
    # 상추: 0.10187
    # 사과: 0.03327
    # 배: 0.04413
    case_year = {
        "배추": 2*12*3,
        "무": 2*12*3,
        "양파": 4*12*3,
        "감자 수미": 2*12*3,
        "대파(일반)": 4*12*3,  # 최근 4년
        "건고추": 2*12*3,
        "깐마늘(국산)": 2*12*3,
        "상추": 4*12*3,
        "사과": 4*12*3,
        "배": 4*12*3,
    }

    for item, item_df in case.items():
        # 20, 21, 22년 데이터만 사용
        target = item_df.tail(case_year[item]).copy()
        case[item] = target

    new_df1 = pd.concat([case['배추'], case['무'], case['양파'], case['감자 수미'], case['대파(일반)']], ignore_index=True)
    new_df2 = pd.concat([case['건고추'], case['깐마늘(국산)'], case['상추'], case['사과'], case['배']], ignore_index=True)

    new_df1.to_csv("./dataset/train/train_1_v1.csv")
    new_df2.to_csv("./dataset/train/train_2_v1.csv")


if __name__ == "__main__":
    add_2017()
