from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 천안 기온 데이터 로드 (2021년도)
data = pd.read_csv('C:/Users/212-10/Desktop/data/gisangcheong/Cheonan temperature.csv', parse_dates=['date'], dayfirst=True, index_col=0, encoding='cp949')
# 헤더값 확인
data.head()

# 인덱스 확인
data.index

# 천안 기온 기간별 데이터 확인
plt.figure(figsize=(16, 9)) # 그래프 사이즈 설정
sns.lineplot(y=data['temperature'], x=data.index)
plt.xlabel('time')
plt.ylabel('temperature')

#분기별 데이터 분리하여 확인
time_steps = [['20210101', '20210301'], 
              ['20210401', '20210601'], 
              ['20210701', '20210901'], 
              ['20211001', '20211201']]

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16, 9)
for i in range(4):
    ax = axes[i//2, i%2]
    df = data.loc[(data.index > time_steps[i][0]) & (data.index < time_steps[i][1])]
    sns.lineplot(y=df['temperature'], x=df.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
plt.tight_layout()
plt.show()

#데이터 정규화 처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 정규화처리 컬럼 지정
scale_cols = ['temperature']
# 정규화처리 데이터 확인
scaled = scaler.fit_transform(data[scale_cols])
scaled

#데이터 프레임 적용
df = pd.DataFrame(scaled, columns=scale_cols)

#학습 및 테스트 데이터 추출
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, df['temperature'], test_size=0.2, random_state=0, shuffle=False)

#학습 데이터 추출 확인
x_train

#학습 데이터 공간 확인
x_train.shape, y_train.shape

#신경망 배치 사이즈
import tensorflow as tf
def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

#WINDOW_SIZE 예측 데이터 생산시 앞데이터 적용수 (앞전 10개 데이터를 기준으로 1개 예측치 생산)
#BATCH_SIZE 모델평가 평가 계수 (학습 1번당 전체데이터/배치수)
WINDOW_SIZE=10
BATCH_SIZE=32

#trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

#데이터 구성 확인
for data in train_data.take(1):
    print(f'dataset(X) composition(batch_size, window_size, feature number): {data[0].shape}')
    print(f'dataset(Y) composition(batch_size, window_size, feature number): {data[1].shape}')

    #LSTM 예측 모델 적용
#라이브러리 tensorflow.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential([
    Conv1D(filters=32, kernel_size=5,
           padding="causal",
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])

loss = Huber()
optimizer = Adam(0.0005) # 일반적으로 딥러닝 모델 학습시에는 0.001 ~ 0.0001 사이의 값을 많이 사용합니다.
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

#손실율 개선 설정
earlystopping = EarlyStopping(monitor='val_loss', patience=10) # patience=10으로 설정하면 검증 손실이 10 epoch 동안 개선되지 않으면 학습을 종료시킵니다.
#기준 체크 포인트 생성
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)

#학습모델 수행
history = model.fit(train_data, validation_data=(test_data), epochs=50, callbacks=[checkpoint, earlystopping])

#예측 데이터 추출
pred = model.predict(test_data)

#예측 데이터 확인
pred

#실데이터 및 예측데이터 표시
#실제 300개 데이터와 예측 데이터 400개 표시
plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[10:310], label='actual') # x축 설정
plt.plot(pred[:400], label='prediction') # x축 설정
plt.legend()
plt.show()

print ("yo")