import numpy as np
import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Путь к файлу в папке SPS_GS
file_path = 'SPS_GS/gs_data.csv'

# Чтение данных из файла CSV, первая строка содержит названия параметров
data = pd.read_csv(file_path, skiprows=1, sep=';')
# Преобразование данных в числа
data = data.apply(lambda row: row.map(lambda x: list(map(float, x.split(';'))) if isinstance(x, str) else x))

# Определение входных данных (X) и выходных данных (y)
X = data.iloc[:, :].drop(columns=[data.columns[0], data.columns[2]]).values  # Все столбцы, кроме первого и третьего, как входные данные
y = data.iloc[:, [0, 2]].values  # Первый и третий столбцы как целевые значения

# # Создание модели
# model = Sequential([
#     Dense(64, input_dim=X.shape[1], activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(2)  # Два выходных нейрона для предсказания чисел в первом и третьем столбцах
# ])

# # Компиляция модели
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Обучение модели
# model.fit(X, y, epochs=200, batch_size=1, verbose=1)

# # Сохранение модели
# model.save('my_model.h5')

# Загрузка сохраненной модели
loaded_model = load_model('my_model.h5')

# Использование модели для предсказания
predictions = loaded_model.predict(X)
print(y)
print(predictions)
# Сохранение графика
plt.figure(figsize=(12, 6))

# Первый столбец
plt.subplot(1, 2, 1)
plt.plot(y[:, 0], label='Real Value 1')
plt.plot(predictions[:, 0], label='Predicted Value 1')
plt.title('Real vs Predicted Value (Column 1)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

# Третий столбец
plt.subplot(1, 2, 2)
plt.plot(y[:, 1], label='Real Value 2')
plt.plot(predictions[:, 1], label='Predicted Value 2')
plt.title('Real vs Predicted Value (Column 3)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

# Сохранение графика в файл
plt.savefig('real_vs_predicted.png')
plt.show()

print("Graph saved as 'real_vs_predicted.png'")
