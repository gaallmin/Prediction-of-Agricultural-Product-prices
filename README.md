# OS, python version

**Windows 10**
**Python 3.11.5**

# Packages

|Package|Version|
|-------|-------|
|pandas             |2.0.3|
|numpy              |1.24.3|
|matplotlib         |3.7.2|
|statsmodels        |0.14.0|

# Usage

1. **Train Preprocessing**:
   - `train_preprocess` 폴더에서 **`preprocess_평균.ipynb`**와 **`preprocess_평년평균.ipynb`**를 실행하여 각각의 CSV 파일을 추출합니다.
   - 이후, **`combine.ipynb`**를 실행하여 두 파일을 합치고, 전처리 및 스무딩 작업을 진행합니다.

2. **Test Preprocessing**:
   - `test_preprocess` 폴더에서 **`test_preprocess.ipynb`**를 실행하여 train에서 적용한 전처리 기법을 test 데이터에도 동일하게 적용합니다.