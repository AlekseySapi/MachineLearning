# Сравним RF и GB модели


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Генерация данных
np.random.seed(42)
data_size = 500

# Признаки: день недели, час и температура
day_of_week = np.random.randint(1, 8, data_size)
hour = np.random.randint(0, 24, data_size)
temperature = np.random.normal(15, 10, data_size)  # Средняя температура около 15°C

# Цель: искусственное потребление электроэнергии с учетом дня, часа и температуры
consumption = (5 * day_of_week +
               3 * hour +
               0.5 * temperature +
               np.random.normal(0, 3, data_size))  # добавляем шум для реалистичности


# Создаём DataFrame
df = pd.DataFrame({
    'day_of_week': day_of_week,
    'hour': hour,
    'temperature': temperature,
    'consumption': consumption
})

# Разделим на обучающую и тестовую выборки
X = df[['day_of_week', 'hour', 'temperature']]
y = df['consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Обучаем модели
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)     # Случайный лес
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42) # Градиентный бустинг

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)


# Предсказание и оценка точности
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
gb_mae = mean_absolute_error(y_test, gb_predictions)

print(f"Средняя абсолютная ошибка (MAE) модели Случайный лес: {rf_mae}")
print(f"Средняя абсолютная ошибка (MAE) модели Градиентный бустинг: {gb_mae}")