import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Путь к файлу в папке SPS_GS
file_path = 'SPS_GS/gs_data.csv'

# Чтение данных из файла CSV, первая строка содержит названия параметров
data = pd.read_csv(file_path, skiprows=1, sep=';')

# Преобразование данных в числа
data = data.apply(lambda row: row.map(lambda x: list(map(float, x.split(';'))) if isinstance(x, str) else x))

# Определение входных данных (X) и выходных данных (y1 и y3)
X = data.iloc[1:, :].drop(columns=[data.columns[0], data.columns[2]]).values  # Все столбцы, кроме первого и третьего, как входные данные
y1 = data.iloc[1:, [0]].values  # Первый столбец как целевое значение
y3 = data.iloc[1:, [2]].values  # Третий столбец как целевое значение

# Нормализация данных
scaler_X = StandardScaler()
scaler_y1 = StandardScaler()
scaler_y3 = StandardScaler()
X = scaler_X.fit_transform(X)
y1 = scaler_y1.fit_transform(y1)
y3 = scaler_y3.fit_transform(y3)

# # Создание модели для предсказания первого столбца
# model1 = Sequential([
#     Dense(128, input_dim=X.shape[1], activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(1)  # Один выходной нейрон для предсказания числа в первом столбце
# ])

# # Компиляция модели для первого столбца
# model1.compile(optimizer='adam', loss='mean_squared_error')

# # Обучение модели для первого столбца
# model1.fit(X, y1, epochs=500, batch_size=10, verbose=1)
# model1.save('model1.h5')

# # Создание модели для предсказания третьего столбца
# model3 = Sequential([
#     Dense(128, input_dim=X.shape[1], activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(1)  # Один выходной нейрон для предсказания числа в третьем столбце
# ])

# # Компиляция модели для третьего столбца
# model3.compile(optimizer='adam', loss='mean_squared_error')

# # Обучение модели для третьего столбца
# model3.fit(X, y3, epochs=500, batch_size=10, verbose=1)
# model3.save('model3.h5')

# Загрузка моделей
loaded_model1 = load_model('model1.h5')
loaded_model3 = load_model('model3.h5')

# Использование моделей для предсказания
predictions1 = loaded_model1.predict(X)
predictions3 = loaded_model3.predict(X)

# Обратная нормализация предсказаний
predictions1 = scaler_y1.inverse_transform(predictions1)
predictions3 = scaler_y3.inverse_transform(predictions3)
y1_true = scaler_y1.inverse_transform(y1)
y3_true = scaler_y3.inverse_transform(y3)

# Построение графиков
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(y1_true, label='Real Value 1')
plt.plot(predictions1, label='Predicted Value 1')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Real vs Predicted Value (Column 1)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y3_true, label='Real Value 3')
plt.plot(predictions3, label='Predicted Value 3')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Real vs Predicted Value (Column 3)')
plt.legend()

plt.tight_layout()
plt.savefig('real_vs_predicted_separate_models.png')
plt.show()