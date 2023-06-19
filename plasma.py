import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from shutil import make_archive
from scipy.stats import skew
from scipy import signal
import joblib
import keras
import os

for i in os.listdir():
    if i[-4:]==".dat" and i != "fil.dat":
        file = i

with open(file, 'r') as f:
    lines = f.readlines()
#print(lines[:10])
# удаляем первую строку
lines.pop(0)

# удаляем последние 4 строки
lines = lines[:-4]

# удаляем 4 пробела в начале каждой строки
lines = [line[4:] if line.startswith('    ') else line for line in lines]

# заменяем двойные пробелы на одинарные
lines = [line.replace('  ', ' ') for line in lines]

with open('fil.dat', 'w') as f:
    f.writelines(lines)

with open('fil.dat', 'r') as f:
    lines = f.readlines()

# удаляем первую строку
lines.pop(0)

# удаляем последние 4 строки
lines = lines[:-4]

# удаляем 4 пробела в начале каждой строки
lines = [line[4:] if line.startswith('    ') else line for line in lines]

# заменяем двойные пробелы на одинарные
lines = [line.replace('  ', ' ') for line in lines]

with open('fil.dat', 'w') as f:
    f.writelines(lines)

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
        if base[i:i + l] == query:
            return i
    return False


# Загрузка всех столбцов из файла
data = pd.read_table("fil.dat", sep=" ", names=["t"] + ["ch{}".format(i) for i in range(1, 300)])

# Предложение пользователю выбрать канал
available_channels = [col for col in data.columns if col != "t" and str(data[col][0]) != 'nan']
print("Доступные каналы:", ', '.join(available_channels))
selected_channel = input("Выберите канал: ")

# Проверка наличия выбранного канала в данных
if selected_channel in available_channels:
    selected_data = data[["t", selected_channel]].astype({"t": "float64"})
    # Продолжайте работу с выбранным каналом (selected_data)
else:
    print("Выбранный канал не найден в данных.")


# plt.figure(figsize=(10, 5))
# sns.set()
# sns.lineplot(x="t", y=selected_channel, data=data)
# plt.savefig("plot.png", dpi=600)
# plt.show()

# Выбираем область графика
start = -np.inf
end = np.inf

x = np.array(data.t[(data.t > start) * (data.t < end)])
y = np.array(data[selected_channel][(data.t > start) * (data.t < end)])

# plt.figure(figsize=(10, 5))
# plt.plot(x, y)
# plt.title("Первоначальный набор данных")
# plt.show()

y_d2 = np.diff(y, n=2)
x_d2 = x[:-2]

b, a = signal.butter(3, 0.4)

y_d2 = signal.filtfilt(b, a, y_d2)

# Просто выбираем в качестве аномалий то, что +- стандартное отклонение. В лоб, но может сработать

m = y_d2.mean()
std = y_d2.std()

# преобразование Фурье и получение массива частот
fft = np.fft.fft(y_d2)
frequency = np.abs(np.fft.fftfreq(len(y_d2)))

# plt.figure(figsize=(10, 5))
colors = ["blue", "red"]
region = 3

# for i in range(len(x_d2) - region):
#     condition = False
#     for k in range(-region, region + 1):
#         condition = condition or ((y_d2[i + k] > m + std) or (y_d2[i + k] < m - std)) and frequency[i] > 0.0
#     if condition:
#         color = colors[1]
#     else:
#         color = colors[0]
#     plt.plot(x_d2[i:i + 2], y_d2[i:i + 2], c=color, alpha=0.7)
# plt.title("Вторая производная и филаменты")
# plt.show()
# plt.figure(figsize=(10, 5))

# plt.show()

y_d2_logic = np.zeros(len(y_d2))

for i in range(3, len(y_d2_logic) - 3):
    condition = False
    for k in range(-region, region + 1):
        condition = condition or ((y_d2[i + k] > m + std) or (y_d2[i + k] < m - std)) \
                    and frequency[i] > 0.03
    if condition:
        y_d2_logic[i] = True

y_f = np.array([y[i] if y_d2_logic[i] else 0 for i in range(len(y_d2_logic))])

# plt.figure(figsize=(10, 5))
# plt.plot(x_d2, y_f)

tolerance = 1  # Чем выше, тем больше шанс получить два филамента на одной картинке
periods = 3  # В среднем количество колебаний на графике, начальный порог
length = 10  # Характеристика длины филамента
sinusoidality = 0.8  # Абсолютная асимметрия
edges = 20  # Сколько точек добавляем слева и справа от филамента.

preprocessed = ";".join(map(str, y_f)).split("0.0;" * tolerance)
preprocessed = [i.split(";") for i in preprocessed if len(i) > 1]
for i in range(len(preprocessed)):
    preprocessed[i] = [float(j) for j in preprocessed[i] if j != ""]
final = np.array([i for i in preprocessed if len(i) > length])

for i in range(len(final)):
    final[i] = np.array(final[i])

filaments = [[], []]

for i in range(len(final)):
    y_ = final[i]

    k = 0
    mean = np.mean(y_)
    for j in range(len(y_) - 1):
        if (y_[j] - mean) * (y_[j + 1] - mean) < 0:
            k += 1

    abs_skewness = np.abs(skew(y_))

    if k > periods * 2 and abs_skewness < sinusoidality:
        r = x_in_y(final[i][:5].tolist(), data[selected_channel].tolist())
        x_id = np.array(list(range(r - edges, r + edges + final[i].shape[0])))
        x_ = data.t[x_id]
        y_ = data[selected_channel][x_id]

        x_smooth = np.linspace(x_.min(), x_.max(), 64)
        y_smooth = interpolate.make_interp_spline(x_, y_)(x_smooth)

        filaments[0].append(x_smooth)
        filaments[1].append(y_smooth)
    k = 0

for i in range(len(filaments[0])):
    filament_t = filaments[0][i]
    filament_f = filaments[1][i]

# print("==========================================")
# print(f"Количество найденных филаментов: {len(filaments[0])}")
# print("==========================================")

neuro_filter = keras.models.load_model("plasma_filaments/neuro_filter.h5")
scaler = joblib.load("plasma_filaments/scaler_for_neuro_filter.pkl")
filtered = neuro_filter.predict(scaler.transform(filaments[1])) > 0.8

count = 0

def create_dataframe():
    data = {
        "Время обнаружения филамента, мс": [],
        "Частота допл. сдвига": [],
        "Длительность филамента, мкс": [],
        "Время между данным и предыдущим, мкс": []
    }
    df = pd.DataFrame(data)
    return df

def add_row(df, detection_time, shift_frequency, filament_duration, time_since_previous):
    row = {
        "Время обнаружения филамента, мс": detection_time,
        "Частота допл. сдвига": shift_frequency,
        "Длительность филамента, мкс": filament_duration,
        "Время между данным и предыдущим, мкс": time_since_previous
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

def calculate_doppler_shift_frequency(signal, sample_rate):
    # Вычисление автокорреляции сигнала
    autocorr = np.correlate(signal, signal, mode='full')

    # Нахождение положительного пика автокорреляции (больше нуля)
    positive_peaks = np.where(autocorr > 0)[0]

    # Нахождение первого положительного пика, отличного от нуля
    peak_index = positive_peaks[0] if positive_peaks.any() else None

    if peak_index is not None:
        # Вычисление частоты Доплеровского сдвига
        frequency_bins = len(signal) * 2  # Количество бинов в автокорреляции
        doppler_shift_frequency = (peak_index - len(signal)) / (2.0 * len(signal)) * sample_rate / frequency_bins
    else:
        doppler_shift_frequency = None

    return doppler_shift_frequency

def subtract_last_detection_time(df, detection_time):
    if len(df) > 0:
        last_detection_time = df.iloc[-1]["Время обнаружения филамента, мс"]
        result = round((detection_time - last_detection_time)*1000)
    else:
        # Handle the case when the DataFrame is empty
        result = None
    return result

# Создание пустой таблицы
info = create_dataframe()

for i in range(len(filaments[0])):
    filament_t = filaments[0][i]
    filament_f = filaments[1][i]

    if filtered[i]:
      # count += 1
      # names = ["Филамент", "не филамент"]
      # c = 0
      # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
      # ax.set_title(f"""{names[c]} на участке [{filament_t.min()}, {filament_t.max()}]""")
      # ax.plot(filament_t, filament_f, color=colors[c])
      # plt.savefig(
      #     f"""plasma_filaments/data/{i} Филамент на участке [{filament_t.min()}, {filament_t.max()}].png""",
      #     dpi=120)
      # plt.show()
      # c = 0
      detection_time = (filament_t.max() - filament_t.min())/2 + filament_t.min()
      shift_frequency = "Wait for updates..."
      filament_duration = round((filament_t.max() - filament_t.min())*1000)
      time_since_previous = subtract_last_detection_time(info, detection_time)
      info = add_row(info, detection_time, shift_frequency, filament_duration, time_since_previous)

info.to_excel(file[:-4] + ".xlsx", index=False)
