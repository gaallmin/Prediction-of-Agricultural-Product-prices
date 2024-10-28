from sys import path
path.append('../')

from tensorflow.keras.models import load_model

from ml_submission import submit
from conv1d_dense import NMAE, create_model

INPUT_SIZE = 9
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
SAVE_FOLDER = './saved_model/'

case_shape = {
    "배추": (INPUT_SIZE, 2),
    "무": (INPUT_SIZE, 4),
    "양파": (INPUT_SIZE, 4),
    "감자 수미": (INPUT_SIZE, 2),
    "대파(일반)": (INPUT_SIZE, 2),
    "건고추": (INPUT_SIZE, 4),
    "깐마늘(국산)": (INPUT_SIZE, 2),
    "상추": (INPUT_SIZE, 2),
    "사과": (INPUT_SIZE, 2),
    "배": (INPUT_SIZE, 2),
}
models = {}
for item in CASE:
    #models[item] = create_model(case_shape[item])
    models[item] = load_model(SAVE_FOLDER + item + '_v2.keras', custom_objects={'NMAE': NMAE}, compile=False)

submit(
    f"../submission/v2(all, 500)_conv2d_dense.csv",
    "../dataset/test",
    "../sample_submission.csv",
    models,
    input_size=9,
    process_method='ewma',
    is_month=True,
    weather = {
        # 리스트는 '품종', '지역', '피쳐' 순
        #'배추': ['가을', 'B', '순 강수량', '순 평균기온'],
        '건고추': ['-', 'E', '순 강수량', '순 평균기온'],
        '무': ['가을', 'B', '순 강수량', '순 평균기온'],
        '양파': ['중만생종', 'K', '순 강수량', '순 평균기온'],
    },
)
