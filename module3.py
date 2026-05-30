import pandas as pd
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon, f_oneway, levene, shapiro, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

results_df = pd.read_csv('results/experiment_results.csv')

models = ['MLP', 'RNN_LSTM', 'CNN']
datasets = ['ETTm1', 'Exchange-Rate', 'Weather', 'Electricity']

dataset_properties = {
    'ETTm1': {'n_features': 7, 'n_samples': 69680, 'stationary': 1, 'seasonal_strength': 0.72},
    'Exchange-Rate': {'n_features': 8, 'n_samples': 7588, 'stationary': 0, 'seasonal_strength': 0.31},
    'Weather': {'n_features': 21, 'n_samples': 52696, 'stationary': 1, 'seasonal_strength': 0.55},
    'Electricity': {'n_features': 321, 'n_samples': 26304, 'stationary': 1, 'seasonal_strength': 0.81}
}

analysis_data = []
for dataset in datasets:
    for model in models:
        row = results_df[(results_df['dataset'] == dataset) & (results_df['model'] == model)]
        if len(row) > 0:
            analysis_data.append({
                'dataset': dataset,
                'model': model,
                'mae': row['mae'].values[0],
                'rmse': row['rmse'].values[0],
                **dataset_properties[dataset]
            })

df = pd.DataFrame(analysis_data)
df = df.drop_duplicates(subset=['dataset', 'model'])

print("\n1. Корреляции Спирмена:")
for model in models:
    df_m = df[df['model'] == model]
    print(f"\n{model}:")
    for prop in ['n_features', 'n_samples', 'seasonal_strength', 'stationary']:
        if df_m[prop].nunique() > 1:
            corr, p = spearmanr(df_m['mae'], df_m[prop])
            print(f"  {prop}: {corr:.6f} (p = {p:.6f})")

print("\n2. Однофакторный ANOVA (модель → MAE):")
mae_by_model = [df[df['model']==m]['mae'].values for m in models]
f_stat, p_anova = f_oneway(*mae_by_model)
print(f"  F = {f_stat:.6f}, p = {p_anova:.6f}")

print("\n3. Проверка допущений ANOVA:")
model_ols = ols('mae ~ C(model)', data=df).fit()
residuals = model_ols.resid
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"  Шапиро-Уилк: W = {shapiro_stat:.6f}, p = {shapiro_p:.6f}")

levene_stat, levene_p = levene(*mae_by_model)
print(f"  Левен: F = {levene_stat:.6f}, p = {levene_p:.6f}")

print("\n4. Тест Крускала-Уоллиса:")
h_stat, p_kruskal = kruskal(*mae_by_model)
print(f"  H = {h_stat:.6f}, p = {p_kruskal:.6f}")

print("\n5. Двухфакторный ANOVA (модель + стационарность):")
model_interaction = ols('mae ~ C(model) * C(stationary)', data=df).fit()
anova_table = sm.stats.anova_lm(model_interaction, typ=2)
print(anova_table.to_string())

print("\n6. Пост-хок Тьюки (Tukey HSD):")
tukey = pairwise_tukeyhsd(df['mae'], df['model'], alpha=0.05)
print(tukey)

print("\n7. Тест Фридмана:")
pivot = df.pivot(index='dataset', columns='model', values='mae')
mae_matrix = [pivot[model].values for model in models]
chi2, p_friedman = friedmanchisquare(*mae_matrix)
print(f"  χ² = {chi2:.6f}, p = {p_friedman:.6f}")

print("\n8. Пост-хок Вилкоксон (парные сравнения):")
pairs = [('MLP', 'RNN_LSTM'), ('MLP', 'CNN'), ('RNN_LSTM', 'CNN')]
for m1, m2 in pairs:
    stat, p_val = wilcoxon(pivot[m1], pivot[m2])
    print(f"  {m1} vs {m2}: W = {stat:.6f}, p = {p_val:.6f}")

