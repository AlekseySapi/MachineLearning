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