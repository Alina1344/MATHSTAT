import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, norm
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Загрузка и разделение данных по группам ===
df = pd.read_csv('m_fish.csv')
group_a = df[df['group'] == 'A']
group_b = df[df['group'] == 'B']

# === 2. Формулировка гипотез ===

# Гипотеза 1 (о конверсии - клики "добавить в корзину"):
# H0 (нулевая гипотеза): доля кликов (конверсий) в группе A = доле в группе B
# H1 (альтернативная гипотеза): доли кликов различаются

# Гипотеза 2 (о времени сессии):
# H0: средняя продолжительность сессии в группе A = средней в группе B
# H1: средние значения различаются

# === 3. Z-тест для пропорций (долей) ===


clicks_a = group_a['clicked_add_to_cart'].sum()
clicks_b = group_b['clicked_add_to_cart'].sum()
n_a = len(group_a)
n_b = len(group_b)

# Доли кликов (конверсии) в каждой группе
p1 = clicks_a / n_a  # доля в A
p2 = clicks_b / n_b  # доля в B

# Применяем Z-тест для сравнения двух независимых долей
# Это важно, чтобы определить: значимо ли отличается конверсия между группами
z_stat, p_val_z = proportions_ztest([clicks_a, clicks_b], [n_a, n_b])

# Вычисляем 95% доверительный интервал для разности долей
diff_prop = p2 - p1
se_prop = np.sqrt(p1*(1-p1)/n_a + p2*(1-p2)/n_b)

# Число 1.96 — это критическое значение z-распределения при уровне значимости 0.05.

ci_low_prop = diff_prop - 1.96 * se_prop
ci_high_prop = diff_prop + 1.96 * se_prop

# === 4. T-тест Стьюдента для среднего времени сессии ===

# Средние значения и стандартные отклонения
mean_a = group_a['session_duration'].mean()
mean_b = group_b['session_duration'].mean()
std_a = group_a['session_duration'].std()
std_b = group_b['session_duration'].std()

# T-тест используется для количественных данных (время — непрерывная переменная)
# Он проверяет: является ли разница между средними значима статистически

t_stat, p_val_t = ttest_ind(group_a['session_duration'], group_b['session_duration'])

# Вычисляем 95% доверительный интервал для разности средних
diff_mean = mean_b - mean_a
se_mean = np.sqrt(std_a**2/n_a + std_b**2/n_b)
ci_low_mean = diff_mean - 1.96 * se_mean
ci_high_mean = diff_mean + 1.96 * se_mean

# === 5. Вывод результатов ===

print("\n=== Гипотеза 1: Различие в доле кликов (конверсии) ===")
print(f"Доля кликов (A): {p1:.4f}, (B): {p2:.4f}")
print(f"Z-статистика = {z_stat:.4f}, p-значение = {p_val_z:.4f}")
print(f"95% ДИ для разности долей: ({ci_low_prop:.4f}, {ci_high_prop:.4f})")

print("\n=== Гипотеза 2: Различие во времени сессии ===")
print(f"Среднее время (A): {mean_a:.2f}, (B): {mean_b:.2f}")
print(f"T-статистика = {t_stat:.4f}, p-значение = {p_val_t:.4f}")
print(f"95% ДИ для разности средних: ({ci_low_mean:.2f}, {ci_high_mean:.2f})")

# === 6. Визуализация результатов ===

# График конверсии
fig, ax = plt.subplots()
ax.bar(['Group A', 'Group B'], [p1, p2], color=['skyblue', 'orange'])
ax.set_title('Доля кликов "Добавить в корзину"')
ax.set_ylabel('Доля кликов')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# График среднего времени сессии
fig, ax = plt.subplots()
ax.bar(['Group A', 'Group B'], [mean_a, mean_b], color=['skyblue', 'orange'])
ax.set_title('Среднее время сессии (в секундах)')
ax.set_ylabel('Среднее значение')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Распределение по группам
plt.figure(figsize=(10, 6))
sns.histplot(group_a['session_duration'], bins=30, kde=True, color='skyblue', label='Group A', stat='density', alpha=0.6)
sns.histplot(group_b['session_duration'], bins=30, kde=True, color='orange', label='Group B', stat='density', alpha=0.6)
plt.title('Распределение длительности сессий: группы A и B')
plt.xlabel('Время сессии (в секундах)')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
