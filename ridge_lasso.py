import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Генерируем случайные данные
np.random.seed(0)
X = pd.DataFrame({
    'Area': np.random.randint(20, 150, 100),
    'Rooms': np.random.randint(1, 5, 100)
})
y = X['Area'] * 3000 + X['Rooms'] * 5000 + np.random.normal(0, 20000, 100)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём и обучаем модели с регуляризацией
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Предсказываем результаты
ridge_predictions = ridge.predict(X_test)
lasso_predictions = lasso.predict(X_test)

# Оцениваем точность
ridge_mae = mean_absolute_error(y_test, ridge_predictions)
lasso_mae = mean_absolute_error(y_test, lasso_predictions)

print(f"Средняя абсолютная ошибка (MAE) для L2-регуляризации (Ridge): {ridge_mae}")
print(f"Средняя абсолютная ошибка (MAE) для L1-регуляризации (Lasso): {lasso_mae}")