import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import keras_tuner as kt
import matplotlib.pyplot as plt

# Путь к файлу в папке SPS_GS
file_path = 'SPS_GS/gs_data.csv'

# Чтение данных из файла CSV, первая строка содержит названия параметров
data = pd.read_csv(file_path, skiprows=1, sep=';')

# Преобразование данных в числа
data = data.apply(lambda row: row.map(lambda x: list(map(float, x.split(';'))) if isinstance(x, str) else x))

# Определение входных данных (X) и выходных данных (y)
X = data.iloc[1:, :].drop(columns=[data.columns[0], data.columns[2]]).values  # Все столбцы, кроме первого и третьего, как входные данные
y = data.iloc[1:, [0, 2]].values  # Первый и третий столбцы как целевые значения

# Нормализация данных
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Функция для создания модели с гиперпараметрами
def build_model(hp):
    model = Sequential()
    model.add(Dense(
        units=hp.Int('units_1', min_value=32, max_value=512, step=32),
        activation='relu',
        input_dim=X.shape[1],
        kernel_regularizer=l2(hp.Float('l2_1', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    ))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(
        units=hp.Int('units_2', min_value=32, max_value=512, step=32),
        activation='relu',
        kernel_regularizer=l2(hp.Float('l2_2', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    ))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(2))  # Два выходных нейрона для предсказания чисел в первом и третьем столбцах
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Создание тюнера
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=100,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Поиск лучших гиперпараметров
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner.search(X, y, epochs=200, validation_split=0.2, callbacks=[early_stopping])

# Получение лучших гиперпараметров
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# # Обучение модели с лучшими гиперпараметрами
# model = tuner.hypermodel.build(best_hps)
# model.save('my_model.h5')
# Загрузка модели
loaded_model = load_model('my_model.h5')
history = loaded_model.fit(X, y, epochs=200, validation_split=0.2, callbacks=[early_stopping])
# Использование модели для предсказания
predictions = loaded_model.predict(X)

# Обратная нормализация предсказаний
predictions = scaler_y.inverse_transform(predictions)
y_true = scaler_y.inverse_transform(y)

# Построение графиков
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(y_true[:, 0], label='Real Value 1')
plt.plot(predictions[:, 0], label='Predicted Value 1')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Real vs Predicted Value (Column 1)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_true[:, 1], label='Real Value 3')
plt.plot(predictions[:, 1], label='Predicted Value 3')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Real vs Predicted Value (Column 3)')
plt.legend()

plt.tight_layout()
plt.savefig('real_vs_predicted_combined_model.png')
plt.show()

# График обучения
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()