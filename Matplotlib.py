# Изучим библиотеку Matplotlib


import matplotlib.pyplot as plt
import pandas as pd


# plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
# plt.show()


data = {
    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'Views': [90, 160, 120, 180, 300, 520, 480]
}
df = pd.DataFrame(data)

# Линейный график просмотров по дням недели
plt.figure(figsize=(10, 6))
plt.plot(df['Day'], df['Views'], marker='o', linestyle='-', color='b')
plt.title('Views per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Views')
plt.grid(True)
plt.show()


# Гистограмма просмотров по дням недели
plt.figure(figsize=(10, 6))
plt.bar(df['Day'], df['Views'], color='c')
plt.title('Distribution of Views by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Views')
plt.show()