import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Путь к файлу в папке SPS_GS
file_path = 'SPS_GS/gs_data.csv'

# Чтение данных из файла CSV, первая строка содержит названия параметров
data = pd.read_csv(file_path, skiprows=1, sep=';')
# Преобразование данных в числа
data = data.apply(lambda row: row.map(lambda x: list(map(float, x.split(';'))) if isinstance(x, str) else x))
# Определение входных данных (X) и выходных данных (y)
X = data.iloc[1:, :].drop(columns=[data.columns[0], data.columns[2]]).values  # Все столбцы, кроме первого и третьего, как входные данные
y = data.iloc[1:, [0, 2]].values  # Первый и третий столбцы как целевые значения
print
# Создание модели
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(2)  # Два выходных нейрона для предсказания чисел в первом и третьем столбцах
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X, y, epochs=200, batch_size=1, verbose=1)
model.save('my_model.h5')
# import os
# print(os.path.exists('trained_model.h5'))  # Проверяем, что файл существует
# loaded_model = load_model('my_model.h5')
# Использование модели для предсказания
predictions = model.predict(X)
print("Predictions:", predictions)
# загрузить в csv