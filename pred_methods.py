# Изучаю Линейную регрессию и тд
# y = w1x + w0
# ...
# Попробовал Дерево решений и Случайный лес (Random Forest)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor       # Попробуем применить Дерево решений
from sklearn.ensemble import RandomForestRegressor      # Применим Случайный лес
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV        # Выбор оптимальных параметров с помощью Grid Search


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


# Создаём модель Градиентного бустинга
xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)


# Используем Grid Search
param_grid = {
    'n_estimators': [50, 90, 100, 120, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7]
}

# Настройка Grid Search с кросс-валидацией (cv=5)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Обучаем модель с Grid Search
grid_search.fit(X_train, y_train)


# Находим лучшие параметры
best_params = grid_search.best_params_

print("Лучшие параметры:", best_params)


# Создаем модель XGBoost с лучшими параметрами
best_xgb_model = xgb.XGBRegressor(**best_params)

# Обучение модели
best_xgb_model.fit(X_train, y_train)

'''
# Создаем и обучаем модель - Случайный лес
model_rf = RandomForestRegressor(n_estimators=120, max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42)
# n_estimators (число деревьев в лесу) - 100-500
# max_depth (максимальная глубина дерева) - 5-30
# min_samples_split (минимальное количество образцов для разделения) - 2-10
# min_samples_leaf (минимальное количество данных в листовом узле) - 1-5

model_rf.fit(X_train, y_train)
'''


# Делаем предсказание на тестовой выборке
y_pred = best_xgb_model.predict(X_test)



# Оценка модели
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)



# Оценка с помощью кросс-валидации
scores = cross_val_score(best_xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')     # 5 фолдов (5-10)

# Средняя ошибка
mean_mse = -scores.mean()
print(f'Средняя квадратичная ошибка (с кросс-валидацией): {mean_mse}')



# Визуализация важности признаков
import matplotlib.pyplot as plt
xgb.plot_importance(best_xgb_model)
plt.show()


# Визуализация зависимости цены от площади
plt.scatter(X['Area'], y, color='blue')
plt.plot(X, best_xgb_model.predict(X), color='red')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()

# Визуализация зависимости цены от количества комнат
plt.scatter(X['Rooms'], y, color='green')
plt.plot(X, best_xgb_model.predict(X), color='red')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.title('Rooms vs Price')
plt.show()


# Визуализация предсказанных и фактических значений
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price (Random Forest)')
plt.title('Actual vs Predicted Price (Random Forest)')
plt.show()