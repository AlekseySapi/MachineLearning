# Изучаю Линейную регрессию и тд
# y = w1x + w0
# ...
# Попробовал Дерево решений..
# Теперь используем Случайный лес (Random Forest)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor       # Попробуем применить Дерево решений
from sklearn.ensemble import RandomForestRegressor      # Применим Случайный лес
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# Генерируем примерные данные с площадями и ценами
data = {
    'Area': [50, 60, 70, 80, 90, 100, 110, 120],
    'Rooms': [1, 1, 1, 1, 1, 2, 2, 2],
    'Price': [3000, 3400, 4000, 4200, 4700, 5700, 6100, 6900]
}

df = pd.DataFrame(data)

# Разделяем данные на входные (X) и выходные (y)
X = df[['Area', 'Rooms']]
y = df['Price']

# Делим данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Создаем и обучаем модель - Случайный лес
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# n_estimators (число деревьев в лесу) - 100-500
# max_depth (максимальная глубина дерева) - 5-30
# min_samples_split (минимальное количество образцов для разделения) - 2-10
# min_samples_leaf (минимальное количество данных в листовом узле) - 1-5

model_rf.fit(X_train, y_train)

# Делаем предсказание на тестовой выборке
y_pred_rf = model_rf.predict(X_test)


# Применение кросс-валидации на модели случайного леса
scores = cross_val_score(model_rf, X, y, cv=5, scoring='neg_mean_squared_error')     # 5 фолдов (5-10)

# Средняя ошибка
mean_mse = -scores.mean()
print(f'Средняя квадратичная ошибка (с кросс-валидацией): {mean_mse}')


# Визуализация зависимости цены от площади
plt.scatter(X['Area'], y, color='blue')
plt.plot(X, model_rf.predict(X), color='red')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()

# Визуализация зависимости цены от количества комнат
plt.scatter(X['Rooms'], y, color='green')
plt.plot(X, model_rf.predict(X), color='red')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.title('Rooms vs Price')
plt.show()


# Визуализация предсказанных и фактических значений
plt.scatter(y_test, y_pred_rf, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price (Random Forest)')
plt.title('Actual vs Predicted Price (Random Forest)')
plt.show()