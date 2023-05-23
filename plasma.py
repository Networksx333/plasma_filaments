# -*- coding: utf-8 -*-
"""Плазма.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15ZT5UyjdGhAtaLeXHy96prx5UX0z1_QX

## Подключаем библиотеки
"""

import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, MaxPooling1D
from tensorflow.keras.layers import Dropout, Activation, Flatten, UpSampling1D
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Input
from tcn import TCN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.signal import resample
from scipy import interpolate
from IPython.display import Audio, display
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import Sequential
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import plot_model
from sklearn.utils import class_weight
from scipy.fft import fft, fftfreq
from shutil import make_archive
from scipy.stats import skew
from scipy import signal
import numpy as np
from scipy.stats import norm
from scipy.stats import kstest
from keras.models import Model
from copy import copy
import joblib
from matplotlib.colors import LogNorm, Normalize
# Importing a libraries for working with a matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Импорт библиотеки для распаковки
import os
import zipfile

"""## TSD

"""

# # data = pd.read_table("plasma_filaments/filaments.dat", sep=" ", names=["t", "f"])
# with open('plasma_filaments/filaments.dat', 'r') as f:
#     lines = f.readlines()
# print(lines[:10])
# # удаляем первую строку
# lines.pop(0)

# # удаляем последние 4 строки
# lines = lines[:-4]

# # удаляем 4 пробела в начале каждой строки
# lines = [line[4:] if line.startswith('    ') else line for line in lines]

# # заменяем двойные пробелы на одинарные
# lines = [line.replace('  ', ' ') for line in lines]

# with open('filaments.dat', 'w') as f:
#     f.writelines(lines)
data = pd.read_table("plasma_filaments/filaments.dat", sep=" ", names=["t", "f"])
data.astype({"t": "float64"})

plt.figure(figsize=(10, 5))
sns.set()
sns.lineplot(x="t", y="f", data=data)
# plt.savefig("plot.png", dpi=600)
plt.show()

# Выбираем область графика
start = -np.inf
end = np.inf

x = np.array(data.t[(data.t>start)*(data.t<end)])
y = np.array(data.f[(data.t>start)*(data.t<end)])

y_d2 = np.diff(y, n=2)
x_d2 = x[:-2]

b, a = signal.butter(3, 0.4)

y_d2 = signal.filtfilt(b, a, y_d2)

# Просто выбираем в качестве аномалий то, что +- стандартное отклонение. Тупо,
# но может сработать

m = y_d2.mean()
std = y_d2.std()

# fft = np.fft.fft(y_d2)
frequency = np.abs(np.fft.fftfreq(len(y_d2)))
# print(frequency)

plt.figure(figsize=(10,5))
plt.plot(x, y)
plt.title("Первоначальный набор данных")
plt.show()
plt.figure(figsize=(10,5))
colors = ["blue", "red"]
region = 3
# for i in range(len(x_d2)-region):
  
#   condition = False
#   for k in range(-region,region+1):
#     condition = condition or ((y_d2[i+k] > m + std) or (y_d2[i+k] < m - std)) \
#                 and frequency[i] > 0.0
#   if condition:
#     color = colors[1]
#   else:
#     color = colors[0]
#   plt.plot(x_d2[i:i+2], y_d2[i:i+2], c=color, alpha=0.7)
# plt.title("Вторая производная и филаменты")
# plt.show()
# plt.figure(figsize=(10,5))

plt.show()

y_d2_logic = np.zeros(len(y_d2))

for i in range(3, len(y_d2_logic)-3):
    condition = False
    for k in range(-region,region+1):
      condition = condition or ((y_d2[i+k] > m + std) or (y_d2[i+k] < m - std)) \
                                and frequency[i] > 0.03
    if condition:
      y_d2_logic[i] = True

y_f = np.array([y[i] if y_d2_logic[i] else 0 for i in range(len(y_d2_logic))])

plt.figure(figsize=(10, 5))
plt.plot(x_d2, y_f)

def x_in_y(query, base):
    """
    The function returns the index of the subsequence in the sequence

    query: list - subsequence
    base: list - sequence
    """
    try:
        l = len(query)
    except TypeError:
        l = 1
        query = type(base)((query,))

    for i in range(len(base)):
        if base[i:i+l] == query:
            return i
    return False

def get_extremums(signal):
    extremums = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            extremums.append(signal[i])
        elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            extremums.append(signal[i])
    return extremums

def check_normality(extremums):
    k2, p = kstest(extremums, 'norm')
    alpha = 0.0005
    if p > alpha:
        return "Распределение экстремумов близко к нормальному"
    else:
        return "Распределение экстремумов не является нормальным"

tolerance = 1 # Чем выше, тем больше шанс получить два филамента 
              # на одной картинке
periods = 3 # В среднем количество колебаний на графике, начальный порог
# extr = 6 # Количество экстремумов
length = 10 # Характеристика длины филамента
sinusoidality = 0.8 # Абсолютная асимметрия
edges = 20 # Сколько точек добавляем слева и справа от филамента.
           # Так можно увидеть окружение филаментов

preprocessed = ";".join(map(str, y_f)).split("0.0;"*tolerance)
preprocessed = [i.split(";") for i in preprocessed if len(i)>1]
for i in range(len(preprocessed)):
  preprocessed[i] = [float(j) for j in preprocessed[i] if j!=""]
final = np.array([i for i in preprocessed if len(i)>length])

for i in range(len(final)):
  final[i] = np.array(final[i])

filaments = [[],[]]

for i in range(len(final)):
  y_ = final[i]
  
  k = 0
  mean = np.mean(y_)
  for j in range(len(y_)-1):
    if (y_[j] - mean)*(y_[j+1] - mean) < 0:
      k += 1


  # extremums = get_extremums(y_)

  # k = len(extremums)
  # simmetry = np.abs(len([j for j in extremums if j<mean]) - 
  #                   len([j for j in extremums if j>mean]))

  abs_skewness = np.abs(skew(y_))

  if k > periods*2 and abs_skewness < sinusoidality:
    r = x_in_y(final[i][:5].tolist(), data.f.tolist())
    x_id = np.array(list(range(r-edges, r+edges+final[i].shape[0])))
    x_ = data.t[x_id]
    y_ = data.f[x_id]

    x_smooth = np.linspace(x_.min(), x_.max(), 64)
    y_smooth = interpolate.make_interp_spline(x_, y_)(x_smooth)

    filaments[0].append(x_smooth)
    filaments[1].append(y_smooth)
  k = 0

count = 0
for i in range(len(filaments[0])):
  filament_t = filaments[0][i]
  filament_f = filaments[1][i]

  # try:
  #   filrn = copy(filr.tolist())
  # except:
  #   pass
  # for i in filrn:
  #     if (i < filament_t.max()) and (i > filament_t.min()):
  #       count += 1
  #       filrn.remove(i)
  #       break
print("==========================================")
print(f"Количество найденных филаментов: {len(filaments[0])}")
# print(f"Количество реальных филаментов: {count}")
print("==========================================")

# with zipfile.ZipFile("yes4.zip", 'r') as zip_file:
#   zip_file.extractall()

# numbers = [int(name.split()[0]) for name in os.listdir("yes4")]

# answers = []
# fil_for_training = []

# for i in range(len(filaments[0])):
#   filament_t = filaments[0][i]
#   filament_f = filaments[1][i]

#   if i in numbers:
#     fil_for_training.append(filament_f)
#     answers.append(1)
#   else:  
#     fil_for_training.append(filament_f)
#     answers.append(0)

# text = ""

# with open("1.txt", "w") as f1: 
#   for i in range(len(answers)):
#     text += ", ".join(map(str, fil_for_training[i])) + ", " + str(answers[i]) + "\n"
#   f1.write(text)

# fil_for_training = np.array(fil_for_training)
# answers = np.array(answers).reshape(-1, 1)

with open("plasma_filaments/data_filter.txt", "r") as f:
  data_filters = f.readlines()
  data_filters = [line.split(", ") for line in data_filters]
  fil_for_training = [line[:-1] for line in data_filters]
  answers = [int(line[-1]) for line in data_filters]
  fil_for_training = np.array(fil_for_training).astype(np.float64)
  answers = np.array(answers).reshape(-1, 1).astype(np.float64)

!rm -rf untitled_project

def find_key_with_text(dictionary, text):
    for key in dictionary.keys():
        if text in key:
            return key
    return None

# f_train, f_test, answ_train, answ_test = train_test_split(fil_for_training, 
#                                                           answers,
#                                                           test_size=0.3)

# scaler = StandardScaler()

# f_train = scaler.fit_transform(f_train)
# f_test = scaler.transform(f_test)
# joblib.dump(scaler, "scaler_for_neuro_filter.pkl")

# class_weights = class_weight.compute_class_weight(class_weight='balanced', 
#                                                   classes=np.unique(answ_train).astype(int).tolist(),
#                                                   y=answ_train.astype(int).reshape(-1).tolist())
# class_weights = dict(zip(np.unique(answ_train), class_weights))

# model = Sequential()
# model.add(Conv1D(64, kernel_size=4, activation='relu', input_shape=(64, 1)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(64, kernel_size=4, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(128, kernel_size=4, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', 
#               metrics=[keras.metrics.Precision()])

# model.fit(f_train, answ_train, epochs=20, batch_size=16, validation_split=0.2,
#         class_weight=class_weights, verbose=1)
# model.evaluate(f_test, answ_test)
# model.save("neuro_filter.h5")

neuro_filter = keras.models.load_model("plasma_filaments/neuro_filter.h5",
                                       custom_objects={'TCN':TCN})
scaler = joblib.load("plasma_filaments/scaler_for_neuro_filter.pkl")

filtered = neuro_filter.predict(scaler.transform(filaments[1])) > 0.8

trajectory = keras.models.load_model("trajectory_model.h5",
                                     custom_objects={'TCN':TCN})

count = 0

for i in range(len(filaments[0])):
  filament_t = filaments[0][i]
  filament_f = filaments[1][i]
  
  if filtered[i] == True:
    count += 1
    names = ["филамент", "не филамент"]
    # c = 0
    # for i in filr:
    #   if (i < filament_t.max()) and (i > filament_t.min()):
    #     c = 1
    #     count += 1
    #     break

    c = 0

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    ax.set_title(f"""{names[c]} на участке 
    [{filament_t.min()}, {filament_t.max()}]""")
    ax.plot(filament_t, filament_f, color=colors[c])
    # extremums = get_extremums(filament_f)
    # sns.boxplot(data=np.abs(filament_f), ax=ax[1])
    # pred = autoencoder.predict(np.array([filament_f]), verbose=0).reshape(-1, 1)
    # error = 100*np.mean(np.abs(pred - filament_f))
    # ax[1].set_title(f"MAE * 100: {error}")
    # ax[1].plot(filament_t, pred)
    # result = check_normality(extremums)
    # print(result)
    plt.savefig(f"""plasma_filaments/data/{i} {names[c]} на участке 
     [{filament_t.min()}, {filament_t.max()}] в 0 разряде.png""", dpi=120)
    plt.show()
    # clear_output(wait=True)
    c = 0

fil_for_training = np.array(fil_for_training)
print(count)
# print(f"Количество найдённых филаментов {count}, по файлу А.Ю. их должно быть {filr.shape[0]} ")

make_archive("data", "zip", "plasma_filaments/data")

"""## Матрица филаментов"""

# Указание названия архива
archive = 'plasma_filaments/M.zip'

# Открытие и распаковка архива
with zipfile.ZipFile(archive, 'r') as zip_file:
  zip_file.extractall()

import seaborn as sns
# Bresenham Line
def bresenham_line(matrix, i1, j1, i2, j2):
    result = []
    dx = abs(j2 - j1)
    dy = abs(i2 - i1)
    sx = 1 if j1 < j2 else -1
    sy = 1 if i1 < i2 else -1
    err = dx - dy
    while True:
        result.append(matrix[i1][j1])
        if j1 == j2 and i1 == i2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            j1 += sx
        if e2 < dx:
            err += dx
            i1 += sy
    return result


# Set the number of files
n = 6000

# Set freq

f = 29*10**9

# Making an array of positions with id
with open("M/positions.txt", "r") as positions:
  lines = positions.readlines()

lines = lines[2:]
lines = [line.replace("\n", "").split(" ") for line in lines]
lines = [[round(float(line[0])), float(line[1]), float(line[2])] 
         for line in lines]
positions = pd.DataFrame(lines, columns=["id", "r", "z"])

# Create list with ids
ids = []
for i in range(n):
  index = str(i)
  name = "M/globus_" + "0"*(4 - len(index)) + index + ".txt"
  with open(name, "r") as fi:
    lines = fi.readlines()
  lines = lines[2:]
  lines = [line.split(" ")[:-1] for line in lines]
  lines = [[float(line[0]), float(line[1])] 
         for line in lines]
  ids.append(lines)

# Making plot of the positions
x = sorted(list(set(positions.r)))
y = sorted(list(set(positions.z)))

# Creating a variable for the matrix
M = np.zeros((len(y), len(x)))

x_, y_ = np.meshgrid(x, y)

t = 0
for r, z in zip(np.hstack(x_), np.hstack(y_)):
    # print(r, z, sep="\n")
    id = positions.loc[(positions["r"] == r) & \
                       (positions["z"] == z), 'id'].iloc[0]
    for j in range(len(ids[id])):
      if abs(ids[id][j][0] - f) < 10:
        M[t//M.shape[1]][t%M.shape[1]] = ids[id][j][1]
    t += 1

M = M - (np.abs(M) > 1)*M
M = np.round(M, 4)

# Make a line
x0 = 0
y0 = 42
x1 = 99
y1 = 42
h = 1

k = 1
for i in range(k):
  y1 += h
  y0 = y1
  sns.set()
  fig, ax = plt.subplots(1, 2, figsize=(13,5))
  sns.heatmap(ax=ax[0], data=M, norm=Normalize())
  ax[0].plot([y0, y1], [x0, x1], color="white", linewidth=2)
  sns.lineplot(ax=ax[1], data=bresenham_line(M, x0, y0, x1, y1))
  # plt.savefig(f"video/{i}.png")
  plt.show()
  clear_output(wait=True)

x0 = 0
y0 = 0
x1 = 99
y1 = 0
h = 1

# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# shape = cv2.imread("video/" + str(1) + '.png').shape
# video = cv2.VideoWriter('video.mp4', fourcc, 3, (shape[1], shape[0]))

# for j in range(23*2+22+23+22):
#     img = cv2.imread("video/" + str(j) + '.png')
#     video.write(img)

# cv2.destroyAllWindows()
# video.release()

# Make dataset
X_M = []
Y_M = []

for y0 in range(33, 60):
  for y1 in range(33, 60):
    arr = bresenham_line(M, 0, y0, 99, y1)
    arr = np.interp(np.linspace(0, len(arr) - 1, 64), np.arange(len(arr)), arr)
    X_M.append(arr)
    Y_M.append([y0, y1])
X_M = np.array(X_M)
Y_M = np.array(Y_M)

# Neural Network

m_train, m_test, answ_m_train, answ_m_test = train_test_split(X_M, 
                                                              Y_M,
                                                          test_size=0.3)

m_scaler = StandardScaler()

m_train = m_scaler.fit_transform(m_train)
m_test = m_scaler.transform(m_test)
joblib.dump(m_scaler, "scaler_for_matrix.pkl")
input_layer = Input(shape=(64, 1))
x = TCN(nb_filters=64, kernel_size=5, dilations=[1, 2, 4, 8], return_sequences=True)(input_layer)
x = TCN(nb_filters=32, kernel_size=5, dilations=[1, 2, 4, 8], return_sequences=True)(x)
x = TCN(nb_filters=16, kernel_size=5, dilations=[1, 2, 4, 8])(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
output_layer = Dense(2, activation='linear')(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(m_train, answ_m_train, epochs=20)
model.evaluate(m_test, answ_m_test)
model.save("trajectory_model.h5")

def preprocess(x):
  x = np.interp(np.linspace(0, len(x) - 1, 64), np.arange(len(x)), x)
  x = m_scaler.transform(np.array([x]))
  return x

up, down = 0, 99
test_object = preprocess(bresenham_line(M, up, 38, down, 55)).reshape(-1, 64)

model.predict(test_object)

