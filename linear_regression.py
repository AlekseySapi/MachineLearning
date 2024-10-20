# Изучаю Линейную регрессию
# y = w1x + w0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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


# Создаем и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Выводим коэффициенты
print(f"Коэффициент наклона: {model.coef_[0]}")
print(f"Смещение (bias): {model.intercept_}")

# Оценим модель по метрике MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")


# Визуализация зависимости цены от площади
plt.scatter(X['Area'], y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()

# Визуализация зависимости цены от количества комнат
plt.scatter(X['Rooms'], y, color='green')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.title('Rooms vs Price')
plt.show()