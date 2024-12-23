# Инженерия признаков — это процесс добавления новых признаков, выбора, преобразования и удаления лишних

import pandas as pd


# Создание DataFrame с заказами
data = {
    'Order ID': [1, 2, 3, 4, 5],
    'Order Date': pd.to_datetime(['2023-10-01', '2023-10-03', '2023-10-06', '2023-10-07', '2023-10-10']),
    'Delivery Date': pd.to_datetime(['2023-10-05', '2023-10-08', '2023-10-09', '2023-10-12', '2023-10-15']),
    'Price': [100, 250, 300, 150, 600],
    'Items': [2, 4, 3, 2, 6]
}

df = pd.DataFrame(data)

# Добавляем признак Delivery Time
df['Delivery Time'] = (df['Delivery Date'] - df['Order Date']).dt.days

# Добавляем признак Day of the Week
df['Day of the Week'] = df['Order Date'].dt.dayofweek  # 0 - Понедельник, 6 - Воскресенье

# Добавляем признак High Value
df['High Value'] = df['Price'] > 200  # True, если цена > 200, иначе False


print(df)
print("=== === ===")


# Создание DataFrame с сотрудниками
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'Salary': [50000, 54000, 51000, 70000]
}
df = pd.DataFrame(data)

# Фильтрация по условию (возраст > 25)
filtered_df = df[df['Age'] > 25]

# Группировка и агрегирование (поиск средней зарплаты)
average_salary = df['Salary'].mean()

print("Фильтрованный DataFrame:\n", filtered_df)
print("Средняя зарплата:", average_salary)