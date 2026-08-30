import os
SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# reduzir paralelismo para melhorar determinismo
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass

# tentar ativar API de determinismo (se disponível na versão do TF)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

import pymongo
import numpy as np
import pandas as pd
import json
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

## ESSE CÓDIGO MANTÉM O DETERMINISMO EM TREINAMENTOS LSTM COM RETRAINING INCREMENTAL
###############################################
# INDIVIDUALMENTE TESTADO 
##########################
# Configurações
# substituir look_back único por uma lista de valores a testar
look_backs = [1, 3, 5, 10, 20, 30, 60, 90]     # ajuste conforme desejado   

initial_epochs = 100                 # épocas do treino inicial
# incremental_epochs = 10              # épocas por atualização incremental (recomendado 1-2)
batch_size = 32
window_size = None                  # se None, usa todo o conjunto de treino como janela; ou defina um número fixo
verbose = 2

# Parâmetros de callbacks
early_stopping_patience = 15
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# MongoDB connection (ajuste a URI se necessário)
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["papers"]

# Função para criar sequências temporais
def create_sequences(data, labels, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(labels[i + look_back])
    return np.array(X), np.array(y)

# Modelo LSTM (saída direta para 3 classes)
def build_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(100, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(100, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='relu')(x)
    output_layer = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# helper global para pesos de classe (usado tanto no laço normal quanto na função abaixo)
def make_class_weight(y, num_classes=3):
    weights = {i: 1.0 for i in range(num_classes)}
    try:
        classes_present = np.unique(y)
        cw = compute_class_weight('balanced', classes=classes_present, y=y)
        for c, w in zip(classes_present, cw):
            weights[int(c)] = float(w)
    except Exception:
        pass
    return weights

def train_and_evaluate(X_train_raw, y_train, X_test_raw, y_test, look_back, desc):
    
    """
    Escalonamento, treino inicial e re‑treinamento incremental.
    `desc` é somente um sufixo para os checkpoints.
    Retorna (accuracy, rmse, executed_epochs).
    """

    train_flat = X_train_raw.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_flat)
    X_train_scaled = scaler.transform(train_flat).reshape(X_train_raw.shape)
    X_test_scaled = scaler.transform(X_test_raw.reshape(-1, 1)).reshape(X_test_raw.shape)

    X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    y_train_oh = to_categorical(y_train, num_classes=3)
    y_test_oh = to_categorical(y_test, num_classes=3)

    class_weights = make_class_weight(y_train, num_classes=3)

    model = build_lstm_model((look_back, 1))

    init_ckpt_path = os.path.join(checkpoint_dir, f"{desc}_lb{look_back}_init_best.h5")
    inc_ckpt_path = os.path.join(checkpoint_dir, f"{desc}_lb{look_back}_inc_best.h5")

    if len(X_train) > 10:
        callbacks_initial = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=init_ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]
    else:
        callbacks_initial = [
            ModelCheckpoint(filepath=init_ckpt_path, monitor='loss', save_best_only=True, verbose=1)
        ]

    history_initial = model.fit(X_train, y_train_oh, epochs=initial_epochs, batch_size=batch_size,
                                validation_split=0.1 if len(X_train) > 10 else 0.0,
                                shuffle=False, verbose=verbose, class_weight=class_weights,
                                callbacks=callbacks_initial)
    executed_initial_epochs = len(history_initial.history.get('loss', [])) if hasattr(history_initial, 'history') else initial_epochs
    if executed_initial_epochs == 0:
        executed_initial_epochs = initial_epochs
    print(f"[{desc}] épocas executadas no treino inicial: {executed_initial_epochs}")

    checkpoint_inc = ModelCheckpoint(filepath=inc_ckpt_path, monitor='loss', save_best_only=True, verbose=0)

    # preparar buffer para janela deslizante
    if window_size is None:
        ws = len(X_train)
    else:
        ws = window_size

    buffer_X = deque(maxlen=ws)
    buffer_y = deque(maxlen=ws)
    start_idx = max(0, len(X_train) - ws)
    for i in range(start_idx, len(X_train)):
        buffer_X.append(X_train[i])
        buffer_y.append(y_train[i])

    predictions = []
    true_labels = []
    for i in range(len(X_test)):
        x_input = np.expand_dims(X_test[i], axis=0)
        pred_prob = model.predict(x_input, verbose=0)
        pred_class = int(np.argmax(pred_prob, axis=1)[0])
        true_class = int(y_test[i])

        predictions.append(pred_class)
        true_labels.append(true_class)

        buffer_X.append(X_test[i])
        buffer_y.append(true_class)

        X_retrain = np.array(buffer_X)
        y_retrain = to_categorical(np.array(buffer_y), num_classes=3)

        try:
            class_weights_retrain = make_class_weight(np.array(buffer_y), num_classes=3)
        except Exception:
            class_weights_retrain = class_weights

        model.fit(X_retrain, y_retrain, epochs=executed_initial_epochs, batch_size=batch_size,
                  shuffle=False, verbose=0, class_weight=class_weights_retrain,
                  callbacks=[checkpoint_inc])

    preds = np.array(predictions)
    trues = np.array(true_labels)
    accuracy = np.mean(preds == trues)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return accuracy, rmse, executed_initial_epochs

# lista de ações permanece igual
stock_tickers = ["PSSA3", "BRFS3"]
years = [2022, 2023, 2024]
period_modes = ['5y', '3y']          # 5 anos ou 3 anos com janelas deslizantes
results = []  # será preenchido abaixo

for stock_ticker in stock_tickers:
    print(f"\n=== Processando {stock_ticker} ===")
    for mode in period_modes:
        for year in years:
            print(f"\n--- modo={mode} ano={year} ---")
            if mode == '5y':
                start_date = f"{year-5}-01-01"
                query_end = pd.to_datetime(f"{year}-12-31")
            else:  # 3‑anos
                start_date = f"{year-3}-01-01"
                if year == 2024:
                    query_end = pd.to_datetime("2024-12-31")
                else:
                    query_end = pd.to_datetime(f"{year+1}-06-30")  # nove meses seguintes

            print(f"Período de busca: {start_date} a {query_end.date()}")
            query_filter = {
                "date": {"$gte": pd.to_datetime(start_date), "$lte": query_end},
                "codneg": stock_ticker
            }
            query_projection = {"_id": 0, "date": 1, "preult": 1}
            mongo_cursor = mongo_collection.find(query_filter, query_projection)
            stock_data = pd.DataFrame(list(mongo_cursor))

            if stock_data.empty:
                print("Nenhum dado encontrado para esse período.")
                continue

            # preparação básica
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.sort_values(by='date', inplace=True)
            stock_data = stock_data[['preult']].rename(columns={"preult": "Close"})
            stock_data['Trend'] = np.where(
                stock_data['Close'].shift(-1) > stock_data['Close'], 2,
                np.where(stock_data['Close'].shift(-1) < stock_data['Close'], 0, 1)
            )
            stock_data.dropna(inplace=True)
            stock_data.reset_index(drop=True, inplace=True)
            close_values = stock_data[['Close']].values

            for look_back in look_backs:
                print(f"\n--- Testando look_back={look_back} ---")
                X_all_raw, y_all = create_sequences(close_values, stock_data['Trend'].values, look_back)
                if len(X_all_raw) == 0:
                    print("Dados insuficientes para o look_back especificado.")
                    continue

                if mode == '5y':
                    train_size = int(len(X_all_raw) * 0.80)
                    X_train_raw, X_test_raw = X_all_raw[:train_size], X_all_raw[train_size:]
                    y_train, y_test = y_all[:train_size], y_all[train_size:]
                    acc, rmse, epochs_exec = train_and_evaluate(
                        X_train_raw, y_train, X_test_raw, y_test, look_back,
                        f"{stock_ticker}_{mode}_{year}"
                    )
                    print(f"Acurácia: {acc:.4f}")
                    results.append({
                        "stock": stock_ticker,
                        "year": year,
                        "mode": mode,
                        "start_date": start_date,
                        "end_date": str(query_end.date()),
                        "look_back": look_back,
                        "accuracy": float(acc),
                        "rmse": rmse,
                        "epochs_executed": epochs_exec
                    })

                else:  # modo 3 anos – dois sub‑períodos de 9 meses
                    dates = stock_data['date'].values[look_back:]
                    train_end = pd.Timestamp(f"{year-1}-12-31")
                    year_start = pd.Timestamp(f"{year}-01-01")
                    test1_end = year_start + pd.offsets.MonthEnd(8)   # fim de set.
                    test2_end = year_start + pd.offsets.MonthEnd(17)  # fim de jun. do ano +1

                    train_mask = dates <= train_end
                    test1_mask = (dates > train_end) & (dates <= test1_end)
                    test2_mask = (dates > test1_end) & (dates <= test2_end)

                    X_train_raw = X_all_raw[train_mask]
                    y_train = y_all[train_mask]
                    X_test1_raw = X_all_raw[test1_mask]
                    y_test1 = y_all[test1_mask]
                    X_test2_raw = X_all_raw[test2_mask]
                    y_test2 = y_all[test2_mask]

                    if len(X_test1_raw) == 0:
                        print("dados insuficientes para primeiro sub-período de teste.")
                        continue

                    acc1, rmse1, epochs1 = train_and_evaluate(
                        X_train_raw, y_train,
                        X_test1_raw, y_test1,
                        look_back,
                        f"{stock_ticker}_{mode}_{year}_1"
                    )

                    acc2 = np.nan
                    rmse2 = np.nan
                    epochs2 = np.nan
                    if len(X_test2_raw) > 0:
                        X_train2 = np.concatenate([X_train_raw, X_test1_raw], axis=0)
                        y_train2 = np.concatenate([y_train, y_test1], axis=0)
                        acc2, rmse2, epochs2 = train_and_evaluate(
                            X_train2, y_train2,
                            X_test2_raw, y_test2,
                            look_back,
                            f"{stock_ticker}_{mode}_{year}_2"
                        )
                    acc_mean = np.nanmean([acc1, acc2])
                    acc_std = np.nanstd([acc1, acc2])

                    print(f"Acurácia 1: {acc1:.4f}")
                    if not np.isnan(acc2):
                        print(f"Acurácia 2: {acc2:.4f}")
                    print(f"Acurácia média: {acc_mean:.4f}")

                    results.append({
                        "stock": stock_ticker,
                        "year": year,
                        "mode": mode,
                        "look_back": look_back,
                        "accuracy1": float(acc1),
                        "accuracy2": float(acc2) if not np.isnan(acc2) else np.nan,
                        "accuracy_mean": float(acc_mean),
                        "accuracy_std": float(acc_std),
                        "rmse1": rmse1,
                        "rmse2": rmse2,
                        "epochs_executed_1": int(epochs1),
                        "epochs_executed_2": int(epochs2) if not np.isnan(epochs2) else np.nan
                    })

# ao final salvar resultados agregados em CSV
results_df = pd.DataFrame(results)
results_path = os.path.join(os.path.dirname(__file__), "results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nTodos os testes concluídos. Resultados salvos em: {results_path}")
