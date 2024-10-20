# Изучаю Линейную регрессию
# y = w1x + w0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor       # Попробуем применить Дерево решений
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D


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


# Создаем и обучаем модель дерева решений
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Делаем предсказание на тестовой выборке
y_pred = model.predict(X_test)

'''
# Выводим коэффициенты
print(f"Коэффициент наклона: {model.coef_[0]}")
print(f"Смещение (bias): {model.intercept_}")
'''

# Оценим модель по метрике MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")


# Визуализация предсказанных и фактических значений
plt.scatter(y_test, y_pred, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (Decision Tree)')
plt.show()

'''
# Создаём 3D-график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Наносим данные
ax.scatter(X['Area'], X['Rooms'], y, color='red')

# Подписи осей
ax.set_xlabel('Area')
ax.set_ylabel('Rooms')
ax.set_zlabel('Price')

plt.title('Area, Rooms vs Price')
plt.show()


# Предсказания модели
y_pred = model.predict(X_test)

# Визуализация предсказанных и реальных цен
plt.scatter(y_test, y_pred, color='purple')

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.show()
'''