import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Загрузка и разделение данных по группам ===
df = pd.read_csv('m_fish.csv')
group_a = df[df['group'] == 'A']
group_b = df[df['group'] == 'B']

# === 2. Формулировка гипотез ===

# Гипотеза 1 (о конверсии - клики "добавить в корзину"):
# H0: доля кликов (конверсий) в группе A = доле в группе B
# H1: доли кликов различаются

# Гипотеза 2 (о времени сессии):
# H0: средняя продолжительность сессии в группе A = средней в группе B
# H1: средние значения различаются

# === 3. Z-тест для пропорций (долей) ===

clicks_a = group_a['clicked_add_to_cart'].sum()
clicks_b = group_b['clicked_add_to_cart'].sum()
n_a = len(group_a)
n_b = len(group_b)

# Доли кликов
p1 = clicks_a / n_a
p2 = clicks_b / n_b

# Z-тест
z_stat, p_val_z = proportions_ztest([clicks_a, clicks_b], [n_a, n_b])

# 95% доверительный интервал
diff_prop = p2 - p1
se_prop = np.sqrt(p1*(1-p1)/n_a + p2*(1-p2)/n_b)
ci_low_prop = diff_prop - 1.96 * se_prop
ci_high_prop = diff_prop + 1.96 * se_prop

# === 4. Хи-квадрат тест независимости ===

# Таблица сопряжённости: [клики, не клики]
no_clicks_a = n_a - clicks_a
no_clicks_b = n_b - clicks_b

contingency_table = np.array([
    [clicks_a, no_clicks_a],
    [clicks_b, no_clicks_b]
])

# Хи-квадрат тест
chi2_stat, p_val_chi2, dof, expected = chi2_contingency(contingency_table)

# === 5. T-тест Стьюдента для среднего времени сессии ===

mean_a = group_a['session_duration'].mean()
mean_b = group_b['session_duration'].mean()
std_a = group_a['session_duration'].std()
std_b = group_b['session_duration'].std()

t_stat, p_val_t = ttest_ind(group_a['session_duration'], group_b['session_duration'])

# 95% доверительный интервал
diff_mean = mean_b - mean_a
se_mean = np.sqrt(std_a**2/n_a + std_b**2/n_b)
ci_low_mean = diff_mean - 1.96 * se_mean
ci_high_mean = diff_mean + 1.96 * se_mean

# === 6. Вывод результатов ===

print("\n=== Гипотеза 1: Различие в доле кликов (конверсии) ===")
print(f"Доля кликов (A): {p1:.4f}, (B): {p2:.4f}")
print(f"Z-статистика = {z_stat:.4f}, p-значение = {p_val_z:.4f}")
print(f"95% ДИ для разности долей: ({ci_low_prop:.4f}, {ci_high_prop:.4f})")

print("\n=== Хи-квадрат тест ===")
print(f"хи^2-статистика: {chi2_stat:.4f}, p-значение: {p_val_chi2:.4f}")
print(f"Ожидаемые значения при H0:\n{expected}")

print("\n=== Гипотеза 2: Различие во времени сессии ===")
print(f"Среднее время (A): {mean_a:.2f}, (B): {mean_b:.2f}")
print(f"T-статистика = {t_stat:.4f}, p-значение = {p_val_t:.4f}")
print(f"95% ДИ для разности средних: ({ci_low_mean:.2f}, {ci_high_mean:.2f})")

# === 7. Визуализация ===

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

# Распределение длительности сессий
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
