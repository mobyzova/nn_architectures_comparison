#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = "data"
OUTPUT_DIR = "processed"
PLOTS_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASETS = [
    ("ETTm1.csv", "ETTm1"),
    ("exchange_rate.csv", "Exchange-Rate"),
    ("weather.csv", "Weather"),
    ("electricity.csv", "Electricity"),
]

WINDOW_SIZE = 24
HORIZON = 1
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2


def create_sequences(data, window_size, horizon=1):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size + horizon - 1])
    return np.array(X), np.array(y)


def evaluate_stationarity(df, name):
    pvalues = []
    for col in df.columns:
        result = adfuller(df[col].dropna())
        pvalues.append(result[1])
    mean_p = np.mean(pvalues)
    stationary_ratio = sum(p < 0.05 for p in pvalues) / len(pvalues)
    is_stationary = stationary_ratio > 0.5
    return bool(is_stationary), float(mean_p), float(stationary_ratio)


def detect_trend(series):
    t = np.arange(len(series))
    corr = np.corrcoef(t, series)[0, 1]
    if abs(corr) > 0.7:
        return "да"
    elif abs(corr) > 0.3:
        return "слабый"
    else:
        return "нет"


def detect_seasonality(series, period=24):
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(series, nlags=period * 2, fft=False)
    threshold = 2 / np.sqrt(len(series))
    if len(acf_vals) > period and abs(acf_vals[period]) > threshold:
        return "да"
    else:
        return "нет"


def save_plots(df_original, train_df, test_df, first_series, name):
    plt.figure(figsize=(12, 5))
    for col in df_original.columns[:5]:
        plt.plot(df_original.index[:500] if hasattr(df_original.index, '__getitem__') else range(500),
                 df_original.iloc[:500][col], label=col)
    plt.title(f"{name} - исходные временные ряды")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_original_series.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    for col in train_df.columns[:5]:
        plt.plot(range(len(train_df)), train_df[col], label=f"train {col}", alpha=0.8)
        plt.plot(range(len(train_df), len(train_df) + len(test_df)),
                 test_df[col], label=f"test {col}", alpha=0.8)
    plt.axvline(x=len(train_df), color='red', linestyle='--', alpha=0.5)
    plt.title(f"{name} - обучающая и тестовая выборки")
    plt.xlabel("Время")
    plt.ylabel("Нормализованное значение")
    plt.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_train_test_split.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_original.iloc[:, 0].dropna(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title(f"{name} - исходное распределение")
    axes[1].hist(train_df.iloc[:, 0], bins=50, alpha=0.7, color='green')
    axes[1].set_title(f"{name} - после нормализации")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plot_acf(first_series, lags=100, alpha=0.05, ax=plt.gca())
    plt.title(f"{name} - автокорреляционная функция")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_acf.png"), dpi=150)
    plt.close()


results_table = []

for filename, ds_name in DATASETS:
    file_path = os.path.join(DATA_DIR, filename)
    print(f"\n{ds_name}")

    df_raw = pd.read_csv(file_path)
    if 'date' in df_raw.columns:
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw.set_index('date', inplace=True)

    df_interp = df_raw.interpolate(method='linear', limit_direction='both')
    df_interp.ffill(inplace=True)
    df_interp.bfill(inplace=True)

    is_stationary, mean_p, stationary_ratio = evaluate_stationarity(df_interp, ds_name)

    first_series = df_interp.iloc[:, 0]
    trend = detect_trend(first_series)
    seasonality = detect_seasonality(first_series, period=24)

    n_total = len(df_interp)
    train_end = int(n_total * TRAIN_RATIO)
    val_end = int(n_total * (TRAIN_RATIO + VAL_RATIO))

    train_raw = df_interp.iloc[:train_end]
    val_raw = df_interp.iloc[train_end:val_end]
    test_raw = df_interp.iloc[val_end:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    train_df = pd.DataFrame(train_scaled, columns=train_raw.columns)
    val_df = pd.DataFrame(val_scaled, columns=val_raw.columns)
    test_df = pd.DataFrame(test_scaled, columns=test_raw.columns)

    X_train, y_train = create_sequences(train_df.values, WINDOW_SIZE, HORIZON)
    X_val, y_val = create_sequences(val_df.values, WINDOW_SIZE, HORIZON)
    X_test, y_test = create_sequences(test_df.values, WINDOW_SIZE, HORIZON)

    np.savez(os.path.join(OUTPUT_DIR, f"{ds_name}_processed.npz"),
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    log_info = {
        'dataset': str(ds_name),
        'window_size': int(WINDOW_SIZE),
        'horizon': int(HORIZON),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'n_features': int(df_interp.shape[1]),
        'n_timesteps': int(len(df_interp)),
        'stationary': is_stationary,
        'stationary_p_value_mean': mean_p,
        'stationary_ratio': stationary_ratio,
        'trend': str(trend),
        'seasonality': str(seasonality),
        'normalization': 'MinMaxScaler [0,1] fit on train',
        'random_seed': int(RANDOM_SEED)
    }
    with open(os.path.join(OUTPUT_DIR, f"{ds_name}_log.json"), 'w', encoding='utf-8') as f:
        json.dump(log_info, f, indent=4, ensure_ascii=False)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{ds_name}_scaler.pkl"))

    save_plots(df_raw, train_df, test_df, first_series, ds_name)

    print(f"Признаков: {df_interp.shape[1]}")
    print(f"Временных шагов: {len(df_interp)}")
    print(
        f"Стационарность: {'да' if is_stationary else 'нет'} (доля стационарных={stationary_ratio:.1%}, p_mean={mean_p:.3f})")
    print(f"Тренд: {trend}")
    print(f"Сезонность (лаг 24): {seasonality}")
    print(f"Train/val/test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"X_train.shape: {X_train.shape}")

    results_table.append({
        "Датасет": ds_name,
        "Признаки": df_interp.shape[1],
        "Шаги": len(df_interp),
        "Стационарность": f"{'да' if is_stationary else 'нет'} (p={mean_p:.3f})",
        "Тренд": trend,
        "Сезонность": seasonality,
        "Train/val/test": f"{len(train_df)}/{len(val_df)}/{len(test_df)}",
        "X_train": str(X_train.shape)
    })

df_results = pd.DataFrame(results_table)
print(df_results.to_string(index=False))

df_results.to_csv(os.path.join(OUTPUT_DIR, "Table1_summary.csv"), index=False, encoding='utf-8-sig')