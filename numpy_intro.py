# Изучаю NumPy

import numpy as np

# Создание массива
array = np.array([1, 2, 3, 4, 5])

# Операции с массивами
print("Массив:", array)
print("Сумма элементов:", np.sum(array))
print("Среднее значение:", np.mean(array))
print("=== ===")


# Задачи

# 1. Создание массива температур от -5 до 30, 7 чисел
temperatures = np.random.randint(-5, 31, 7)
print("Температуры за неделю:", temperatures)

# 2. Средняя температура
mean_temp = round(np.mean(temperatures), 1)     # Округлил до 1 знака после запятой
print("Средняя температура:", mean_temp)

# 3. Медианная температура
median_temp = np.median(temperatures)
print("Медианная температура:", median_temp)

# 4. Дни с температурой выше средней
higher_than_avg = temperatures[temperatures > mean_temp]
print("Дни с температурой выше средней:", higher_than_avg)

# 5. Увеличение температуры на 2 градуса
adjusted_temperatures = temperatures + 2
print("Температуры после увеличения на 2 градуса:", adjusted_temperatures)