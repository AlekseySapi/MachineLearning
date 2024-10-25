import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Данные о кофе и продуктивности
X = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([5, 70, 90, 30, 40, 10, 15])

# Модель XGBoost
model = XGBRegressor(n_estimators=120)
model.fit(X, y)

# Предсказания модели
X_test = np.linspace(0, 6, 100).reshape(-1, 1)  # Генерация точек для предсказания
y_pred = model.predict(X_test)

# Визуализация
plt.figure(figsize=(10, 6))     # Размер графика в дюймах
plt.scatter(X, y, color='red', label='Исходные данные', zorder=5)  # Исходные данные
plt.plot(X_test, y_pred, color='blue', label='Предсказания модели')  # Линия модели
plt.xlabel('Cups of Coffee')
plt.ylabel('Productivity Level')
plt.title('Cups of Coffee vs Productivity')
plt.legend()
plt.grid(True)
plt.show()