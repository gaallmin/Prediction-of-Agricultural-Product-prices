from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../open/train.csv')
test = pd.read_csv('../open/test.csv')

data = train.rename(columns={'timestamp': 'ds', "price": 'y'})

data = data[['ID', 'ds', 'y']]
data['ID'] = data['ID'].str.replace(r'_\d{8}$', '',regex=True)

RANDOM_SEED = 990313
np.random.seed(RANDOM_SEED)

def ph_train(df):
    pred_list = []

    for code in df['ID'].unique():

        d = df[df['ID'] == code].reset_index().drop(['ID'], axis=1).sort_values('ds')
        model = Prophet()

        print(d)
        model.fit(d)

        future = pd.DataFrame()
        future['ds'] = pd.date_range(start='2023-03-04', periods=28, freq='D')
        forecast = model.predict(future)
        print(forecast)
        pred_y = forecast['yhat'].values
        pred_code = [str(code)] * len(pred_y)

        for y_val, id_val in zip(pred_y, pred_code):
            pred_list.append({'ID': id_val, 'y': y_val})

    pred = pd.DataFrame(pred_list)
    return pred

pred = ph_train(data)

print(pred)
