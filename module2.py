import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_DIR = "processed"
OUTPUT_DIR = "results"
PLOTS_DIR = "plots"
MODELS_DIR = "models"
SUMMARIES_DIR = "model_summaries"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

DATASETS = ["ETTm1", "Exchange-Rate", "Weather", "Electricity"]

HYPERPARAMS = {
    'window_size': 24,
    'horizon': 1,
    'n_epochs': 100,
    'batch_size': 64,
    'learning_rate': 0.001,
    'patience': 10,
    'n_units': 64,
    'n_layers': 2,
    'dropout_rate': 0.2,
    'loss': 'mse',
    'optimizer': 'adam'
}

def build_mlp(input_shape, hyperparams):
    model = Sequential(name="MLP")
    model.add(Input(shape=input_shape))
    model.add(Flatten())

    for i in range(hyperparams['n_layers']):
        model.add(Dense(hyperparams['n_units'], activation='relu'))
        model.add(Dropout(hyperparams['dropout_rate']))

    model.add(Dense(input_shape[1], activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=hyperparams['learning_rate']),
        loss=hyperparams['loss'],
        metrics=['mae']
    )
    return model

def build_rnn(input_shape, hyperparams):
    model = Sequential(name="RNN_LSTM")
    model.add(Input(shape=input_shape))

    for i in range(hyperparams['n_layers'] - 1):
        model.add(LSTM(hyperparams['n_units'], return_sequences=True))
        model.add(Dropout(hyperparams['dropout_rate']))

    model.add(LSTM(hyperparams['n_units'], return_sequences=False))
    model.add(Dropout(hyperparams['dropout_rate']))
    model.add(Dense(input_shape[1], activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=hyperparams['learning_rate']),
        loss=hyperparams['loss'],
        metrics=['mae']
    )
    return model

def build_cnn(input_shape, hyperparams):
    model = Sequential(name="CNN")
    model.add(Input(shape=input_shape))

    for i in range(hyperparams['n_layers']):
        model.add(Conv1D(filters=hyperparams['n_units'], kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hyperparams['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(hyperparams['n_units'], activation='relu'))
    model.add(Dropout(hyperparams['dropout_rate']))
    model.add(Dense(input_shape[1], activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=hyperparams['learning_rate']),
        loss=hyperparams['loss'],
        metrics=['mae']
    )
    return model

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=hyperparams['patience'], restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=hyperparams['batch_size'],
        epochs=hyperparams['n_epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time

    y_pred = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))

    return history, train_time, mae, rmse, y_pred

def plot_predictions(y_test, y_pred, dataset_name, model_name, output_path, n_samples=200):
    plt.figure(figsize=(12, 5))
    y_test_first = y_test[:n_samples, 0]
    y_pred_first = y_pred[:n_samples, 0]

    plt.plot(y_test_first, label='Реальные значения', color='blue', alpha=0.7)
    plt.plot(y_pred_first, label='Прогноз', color='red', alpha=0.7, linestyle='--')
    plt.title(f'{dataset_name} - {model_name}: прогноз vs реальность (1-й признак)')
    plt.xlabel('Временной шаг')
    plt.ylabel('Нормализованное значение')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{dataset_name}_{model_name}_predictions.png'), dpi=150)
    plt.close()

def plot_training_history(history, dataset_name, model_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Обучающая выборка')
    axes[0].plot(history.history['val_loss'], label='Валидационная выборка')
    axes[0].set_title('Функция потерь (MSE)')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['mae'], label='Обучающая выборка')
    axes[1].plot(history.history['val_mae'], label='Валидационная выборка')
    axes[1].set_title('Средняя абсолютная ошибка (MAE)')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('MAE')
    axes[1].legend()

    plt.suptitle(f'{dataset_name} - {model_name}: динамика обучения')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{dataset_name}_{model_name}_history.png'), dpi=150)
    plt.close()

def plot_comparison_bar(results_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = results_df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25

    for i, model in enumerate(['MLP', 'RNN_LSTM', 'CNN']):
        model_data = results_df[results_df['model'] == model]
        maes = [model_data[model_data['dataset'] == d]['mae'].values[0] for d in datasets]
        axes[0].bar(x + i * width, maes, width, label=model)

    axes[0].set_xlabel('Датасет')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Сравнение моделей по MAE')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()

    for i, model in enumerate(['MLP', 'RNN_LSTM', 'CNN']):
        model_data = results_df[results_df['model'] == model]
        rmses = [model_data[model_data['dataset'] == d]['rmse'].values[0] for d in datasets]
        axes[1].bar(x + i * width, rmses, width, label=model)

    axes[1].set_xlabel('Датасет')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Сравнение моделей по RMSE')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(datasets)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'models_comparison.png'), dpi=150)
    plt.close()

def plot_time_comparison(results_df, output_path):
    plt.figure(figsize=(10, 6))

    datasets = results_df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25

    for i, model in enumerate(['MLP', 'RNN_LSTM', 'CNN']):
        model_data = results_df[results_df['model'] == model]
        times = [model_data[model_data['dataset'] == d]['train_time'].values[0] for d in datasets]
        plt.bar(x + i * width, times, width, label=model)

    plt.xlabel('Датасет')
    plt.ylabel('Время обучения (секунды)')
    plt.title('Сравнение вычислительной эффективности')
    plt.xticks(x + width, datasets)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'time_comparison.png'), dpi=150)
    plt.close()

def main():
    print(f"\nГиперпараметры эксперимента:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")

    all_results = []

    for dataset_name in DATASETS:
        print(f"Датасет: {dataset_name}")

        data_path = os.path.join(DATA_DIR, f"{dataset_name}_processed.npz")
        data = np.load(data_path)

        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']

        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"\nВходная размерность: {input_shape}")
        print(f"Обучающая выборка: {X_train.shape[0]} примеров")
        print(f"Валидационная выборка: {X_val.shape[0]} примеров")
        print(f"Тестовая выборка: {X_test.shape[0]} примеров")

        models = {
            'MLP': build_mlp(input_shape, HYPERPARAMS),
            'RNN_LSTM': build_rnn(input_shape, HYPERPARAMS),
            'CNN': build_cnn(input_shape, HYPERPARAMS)
        }

        for model_name, model in models.items():
            print(f"\n{'-' * 60}")
            print(f"Модель: {model_name}")
            print(f"{'-' * 60}")

            summary_file = os.path.join(SUMMARIES_DIR, f"{dataset_name}_{model_name}_summary.txt")
            tee = Tee(summary_file)
            model.summary()
            tee.close()

            print(f"\nОбучение {model_name}...")

            history, train_time, mae, rmse, y_pred = evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test, HYPERPARAMS
            )

            n_params = model.count_params()

            print(f"Время обучения: {train_time:.2f} с")
            print(f"Количество параметров: {n_params}")
            print(f"MAE: {mae:.6f}")
            print(f"RMSE: {rmse:.6f}")

            model.save(os.path.join(MODELS_DIR, f"{dataset_name}_{model_name}.h5"))

            plot_predictions(y_test, y_pred, dataset_name, model_name, PLOTS_DIR)
            plot_training_history(history, dataset_name, model_name, PLOTS_DIR)

            all_results.append({
                'dataset': dataset_name,
                'model': model_name,
                'mae': mae,
                'rmse': rmse,
                'train_time': train_time,
                'n_params': n_params,
                'n_epochs': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'val_final_loss': history.history['val_loss'][-1]
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "experiment_results.csv"), index=False, encoding='utf-8-sig')

    pivot_mae = results_df.pivot(index='dataset', columns='model', values='mae')
    pivot_rmse = results_df.pivot(index='dataset', columns='model', values='rmse')
    pivot_time = results_df.pivot(index='dataset', columns='model', values='train_time')
    pivot_params = results_df.pivot(index='dataset', columns='model', values='n_params')

    print("Результаты эксперимента")

    print("\nСравнение по MAE:")
    print(pivot_mae.to_string())

    print("\nСравнение по RMSE:")
    print(pivot_rmse.to_string())

    print("\nСравнение по времени обучения (секунды):")
    print(pivot_time.to_string())

    print("\nСравнение по количеству параметров:")
    print(pivot_params.to_string())

    plot_comparison_bar(results_df, PLOTS_DIR)
    plot_time_comparison(results_df, PLOTS_DIR)

    pivot_mae.to_csv(os.path.join(OUTPUT_DIR, "comparison_mae.csv"))
    pivot_rmse.to_csv(os.path.join(OUTPUT_DIR, "comparison_rmse.csv"))
    pivot_time.to_csv(os.path.join(OUTPUT_DIR, "comparison_time.csv"))
    pivot_params.to_csv(os.path.join(OUTPUT_DIR, "comparison_params.csv"))

if __name__ == "__main__":
    main()