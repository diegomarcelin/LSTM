import os
SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# Opcional: forçar CPU (descomentar para testes em CPU)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
look_backs = [1, 3, 5, 10, 20, 30]     # ajuste conforme desejado
initial_epochs = 100                 # épocas do treino inicial
batch_size = 32
window_size = None                  # se None, usa todo o conjunto de treino como janela; ou defina um número fixo
verbose = 1

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

# Função para calcular class weights robusta (garante chaves 0, 1, 2)
def make_class_weight(y, num_classes=3):
    """Calcula class weights garantindo que todas as classes têm uma chave no dict"""
    weights = {i: 1.0 for i in range(num_classes)}  # inicializa com peso neutro
    try:
        classes_present = np.unique(y)
        cw = compute_class_weight('balanced', classes=classes_present, y=y)
        for c, w in zip(classes_present, cw):
            weights[int(c)] = float(w)
    except Exception:
        pass  # mantém pesos neutros em caso de erro
    return weights

# Modelo LSTM (saída direta para 3 classes)
def build_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(100, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(100, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Configurações de períodos para teste ---2017---
# ("2017-09-18", "2017-12-31") -> 3 meses de dados (90 dias de treinamento e 18 dias de teste)
# ("2017-04-20", "2017-12-31") -> 6 meses de dados (180 dias de treinamento e 36 dias de teste)
# ("2016-09-15", "2017-12-31") -> 1 ano de dados (12 meses de treinamento e 2.5 de teste)
# ("2014-03-01", "2017-12-31") -> 3 anos de dados (36 meses de treinamento e 9 de teste)
# ("2012-01-01", "2017-12-31") -> 5 anos de dados (60 meses de treinamento e 12 de teste)

# Configurações de períodos para teste ---2018---
# ("2017-09-18", "2017-12-31") -> 3 meses de dados (90 dias de treinamento e 18 dias de teste)
# ("2017-04-20", "2017-12-31") -> 6 meses de dados (180 dias de treinamento e 36 dias de teste)
# ("2017-01-01", "2018-03-16") -> 1 ano de dados (12 meses de treinamento e 2.5 de teste)
# ("2015-01-01", "2018-09-30") -> 3 anos de dados (36 meses de treinamento e 9 de teste)
# ("2013-01-01", "2018-12-31") -> 5 anos de dados (60 meses de treinamento e 12 de teste)

# Períodos
stock_tickers = ["WEGE3"]  #"BRFS3", "GGBR4","CIEL3", "ITSA4","ALPA4"]

# Lista de anos de validação
validation_years = [2021]

# resultados: lista de dicionários (um por teste)
results = []

# map para armazenar quantas épocas foram efetivamente executadas no treino inicial por look_back
executed_epochs_map = {}

for stock_ticker in stock_tickers:
    print(f"\n=== Processando {stock_ticker} ===")
    
    # iterar sobre cada ano de validação
    for validation_year in validation_years:
        validation_period_start = pd.to_datetime(f"{validation_year}-01-01")
        validation_period_end = pd.to_datetime(f"{validation_year}-12-31")
        
        # configuração dos 3 casos: 3m, 6m, 1y
        sliding_cases = [
            {"name": "3m", "train_months": 3, "test_months": 1},
            # {"name": "6m", "train_months": 6, "test_months": 2},
            # {"name": "1y", "train_months": 12, "test_months": 3},
        ]
        
        # ========================================
        # Para cada CASO (3m, 6m, 1y)
        # ========================================
        for case in sliding_cases:
            train_months = case['train_months']
            test_months = case['test_months']
            case_name = case['name']
            
            # Calcular período de TREINO baseado em meses
            if case_name == "6m":
                # Para 6m: treino de junho a dezembro do ano anterior
                train_start_date = pd.to_datetime(f"{validation_year-1}-06-01")
                train_end_date = pd.to_datetime(f"{validation_year-1}-12-31")
            elif case_name == "3m":
                # Para 3m: treino de outubro a dezembro do ano anterior
                train_start_date = pd.to_datetime(f"{validation_year-1}-10-01")
                train_end_date = pd.to_datetime(f"{validation_year-1}-12-31")
            else:  # 1y
                # Para 1y: treino do ano anterior completo
                train_start_date = pd.to_datetime(f"{validation_year-1}-01-01")
                train_end_date = pd.to_datetime(f"{validation_year-1}-12-31")
            
            # Calcular dias aproximados para teste (proporcional aos meses)
            test_days = int(test_months * 30.44)  # média de dias por mês
            
            train_start_date_str = train_start_date.strftime("%Y-%m-%d")
            train_end_date_str = train_end_date.strftime("%Y-%m-%d")
            validation_period_start_str = validation_period_start.strftime("%Y-%m-%d")
            validation_period_end_str = validation_period_end.strftime("%Y-%m-%d")
            
            print(f"\n{'='*70}")
            print(f"Caso: {case_name} | Validação: {validation_year}")
            print(f"Treino (ano anterior): {train_start_date_str} a {train_end_date_str} ({train_months} meses)")
            print(f"Teste (validação): {validation_period_start_str} a {validation_period_end_str} ({test_months} meses ~ {test_days} dias por bloco)")
            print(f"{'='*70}")

            # ========================================
            # 1. BUSCAR DADOS DE TREINO (ano anterior)
            # ========================================
            query_filter = {
                "date": {"$gte": train_start_date, "$lte": train_end_date},
                "codneg": stock_ticker
            }
            query_projection = {"_id": 0, "date": 1, "preult": 1}
            mongo_cursor = mongo_collection.find(query_filter, query_projection)
            stock_data = pd.DataFrame(list(mongo_cursor))

            if stock_data.empty:
                print(f"  [ERRO] Nenhum dado de treino encontrado para {case_name}.")
                continue

            # Preparação dos dados de treino
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.sort_values(by='date', inplace=True)
            stock_data = stock_data[['preult']].rename(columns={"preult": "Close"})

            # Criar rótulos de tendência: 0 descer, 1 estável, 2 subir
            stock_data['Trend'] = np.where(
                stock_data['Close'].shift(-1) > stock_data['Close'], 2,
                np.where(stock_data['Close'].shift(-1) < stock_data['Close'], 0, 1)
            )
            stock_data.dropna(inplace=True)
            stock_data.reset_index(drop=True, inplace=True)

            close_values = stock_data[['Close']].values  # formato (n,1)
            print(f"  [OK] Dados de treino carregados: {len(stock_data)} registros")

            # ========================================
            # 2. BUSCAR DADOS DE VALIDAÇÃO (ano atual)
            # ========================================
            query_val = {
                "date": {"$gte": validation_period_start, "$lte": validation_period_end},
                "codneg": stock_ticker
            }
            mongo_cursor_val = mongo_collection.find(query_val, query_projection)
            val_df = pd.DataFrame(list(mongo_cursor_val))

            if val_df.empty:
                print(f"  [ERRO] Nenhum dado de validação encontrado para {case_name} em {validation_year}.")
                continue

            val_df['date'] = pd.to_datetime(val_df['date'])
            val_df.sort_values(by='date', inplace=True)
            val_df = val_df[['date', 'preult']].rename(columns={"preult": "Close"})
            
            val_df['Trend'] = np.where(
                val_df['Close'].shift(-1) > val_df['Close'], 2,
                np.where(val_df['Close'].shift(-1) < val_df['Close'], 0, 1)
            )
            val_df.dropna(inplace=True)
            val_df.reset_index(drop=True, inplace=True)

            val_close_values = val_df[['Close']].values
            print(f"  [OK] Dados de validação carregados: {len(val_df)} registros")

            # ========================================
            # 3. PARA CADA LOOK_BACK
            # ========================================
            for look_back in look_backs:
                print(f"\n  >>> look_back={look_back} | {case_name}")

                # Criar sequências no conjunto de treino
                X_train_raw, y_train = create_sequences(close_values, stock_data['Trend'].values, look_back)
                if len(X_train_raw) == 0:
                    print(f"      Dados insuficientes para treino com look_back={look_back}.")
                    continue
                
                # Armazenar número REAL de sequências de treino (não dias, mas sequências)
                num_train_seqs = len(X_train_raw)

                # Escalar dados de treino (fit apenas no treino)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(X_train_raw.reshape(-1, 1))
                X_train_scaled = scaler.transform(X_train_raw.reshape(-1, 1)).reshape(X_train_raw.shape)
                X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
                y_train_oh = to_categorical(y_train, num_classes=3)

                # Class weights para treino
                class_weights = make_class_weight(y_train, num_classes=3)

                # Construir modelo LSTM
                model = build_lstm_model((look_back, 1))

                # Checkpoint path
                init_ckpt_path = os.path.join(checkpoint_dir, f"{stock_ticker}_{train_start_date_str}_{train_end_date_str}_lb{look_back}_init_best.h5")

                if len(X_train) > 10:
                    callbacks_initial = [
                        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=0),
                        ModelCheckpoint(filepath=init_ckpt_path, monitor='val_loss', save_best_only=True, verbose=0)
                    ]
                else:
                    callbacks_initial = [
                        ModelCheckpoint(filepath=init_ckpt_path, monitor='loss', save_best_only=True, verbose=0)
                    ]

                # Treino inicial
                history_initial = model.fit(X_train, y_train_oh, epochs=initial_epochs, batch_size=batch_size,
                                            validation_split=0.1 if len(X_train) > 10 else 0.0,
                                            shuffle=False, verbose=0, class_weight=class_weights,
                                            callbacks=callbacks_initial)
                
                executed_initial_epochs = len(history_initial.history.get('loss', [])) if hasattr(history_initial, 'history') else initial_epochs
                if executed_initial_epochs == 0:
                    executed_initial_epochs = initial_epochs

                executed_epochs_map[look_back] = executed_initial_epochs
                print(f"      Treino inicial: {executed_initial_epochs} épocas")

                # ========================================
                # 4. SLIDING EVALUATION NO ANO DE VALIDAÇÃO
                # ========================================
                
                # Combinar dados de treino + validação para criar sequências
                combined_close = np.vstack([close_values, val_close_values])
                combined_trend = np.concatenate([stock_data['Trend'].values, val_df['Trend'].values])
                
                X_comb, y_comb = create_sequences(combined_close, combined_trend, look_back)
                if len(X_comb) == 0:
                    print(f"      Dados insuficientes para sliding evaluation com look_back={look_back}.")
                    continue

                # Índice da primeira sequência cuja label pertence ao ano de validação
                train_seq_end = len(close_values) - look_back
                
                # Começar no primeiro rótulo de validação
                idx = max(train_seq_end, 0)
                block_idx = 0
                block_accuracies = []

                print(f"      Iniciando sliding evaluation... (blocos de {test_days} dias)")

                # ========================================
                # Loop de blocos deslizantes
                # ========================================
                while idx + test_days <= len(X_comb):
                    # Usar número real de sequências de treino disponíveis
                    train_seqs = min(num_train_seqs, idx)
                    if train_seqs <= 0:
                        break
                    
                    # Verificar se há dados de treino suficientes
                    if idx - train_seqs < 0:
                        break

                    X_train_slide = X_comb[idx - train_seqs: idx]
                    y_train_slide = y_comb[idx - train_seqs: idx]
                    X_test_slide = X_comb[idx: idx + test_days]
                    y_test_slide = y_comb[idx: idx + test_days]

                    # Escalar usando scaler ajustado nos dados de treino do bloco
                    scaler_slide = MinMaxScaler(feature_range=(0, 1))
                    scaler_slide.fit(X_train_slide.reshape(-1, 1))
                    X_train_slide_scaled = scaler_slide.transform(X_train_slide.reshape(-1, 1)).reshape(X_train_slide.shape)
                    X_test_slide_scaled = scaler_slide.transform(X_test_slide.reshape(-1, 1)).reshape(X_test_slide.shape)

                    X_train_slide_model = X_train_slide_scaled.reshape((X_train_slide_scaled.shape[0], X_train_slide_scaled.shape[1], 1))
                    X_test_slide_model = X_test_slide_scaled.reshape((X_test_slide_scaled.shape[0], X_test_slide_scaled.shape[1], 1))

                    # Treinar modelo no bloco de treino deslizante
                    y_train_slide_oh = to_categorical(y_train_slide, num_classes=3)
                    class_weights_slide = make_class_weight(y_train_slide, num_classes=3)
                    epochs_to_use = executed_epochs_map.get(look_back, initial_epochs)

                    # Construir e treinar novo modelo para este bloco
                    model_slide = build_lstm_model((look_back, 1))
                    if len(X_train_slide_model) > 10:
                        callbacks_slide = [
                            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=0)
                        ]
                        validation_split_slide = 0.1
                    else:
                        callbacks_slide = [
                            EarlyStopping(monitor='loss', patience=max(1, early_stopping_patience // 3), restore_best_weights=True, verbose=0)
                        ]
                        validation_split_slide = 0.0

                    history_slide = model_slide.fit(
                        X_train_slide_model,
                        y_train_slide_oh,
                        epochs=epochs_to_use,
                        batch_size=batch_size,
                        validation_split=validation_split_slide,
                        shuffle=False,
                        verbose=0,
                        class_weight=class_weights_slide,
                        callbacks=callbacks_slide
                    )

                    executed_slide_epochs = len(history_slide.history.get('loss', [])) if hasattr(history_slide, 'history') else epochs_to_use
                    if executed_slide_epochs == 0:
                        executed_slide_epochs = epochs_to_use

                    # Prever no bloco de teste
                    if X_test_slide_model.shape[0] > 0:
                        pred_probs = model_slide.predict(X_test_slide_model, verbose=0)
                        pred_classes = np.argmax(pred_probs, axis=1).astype(int)
                        acc_block = float(np.mean(pred_classes == y_test_slide))
                    else:
                        acc_block = 0.0

                    block_accuracies.append(acc_block)

                    # Tentar calcular datas do bloco (se houver índice válido em val_df)
                    val_idx_start = idx - train_seq_end
                    val_idx_end = idx - train_seq_end + test_days - 1
                    if 0 <= val_idx_start < len(val_df) and 0 <= val_idx_end < len(val_df):
                        block_start_date = val_df['date'].iloc[val_idx_start].date()
                        block_end_date = val_df['date'].iloc[val_idx_end].date()
                    else:
                        block_start_date = "unknown"
                        block_end_date = "unknown"

                    print(f"        Bloco {block_idx}: {block_start_date} → {block_end_date} | acc={acc_block:.4f} (n={len(y_test_slide)})")

                    # Registrar resultado do bloco
                    results.append({
                        "stock": stock_ticker,
                        "train_start": train_start_date_str,
                        "train_end": train_end_date_str,
                        "validation_year": validation_year,
                        "look_back": look_back,
                        "case": case_name,
                        "train_months": train_months,
                        "test_months": test_months,
                        "block_idx": block_idx,
                        "block_start": str(block_start_date),
                        "block_end": str(block_end_date),
                        "block_accuracy": acc_block,
                        "block_size": int(len(y_test_slide)),
                        "executed_epochs": int(executed_slide_epochs),
                        "max_epochs_from_initial": int(epochs_to_use),
                    })

                    idx += test_days
                    block_idx += 1

                # ========================================
                # Calcular média e desvio padrão anuais
                # ========================================
                if block_accuracies:
                    mean_acc = float(np.mean(block_accuracies))
                    std_acc = float(np.std(block_accuracies))
                    
                    results.append({
                        "stock": stock_ticker,
                        "train_start": train_start_date_str,
                        "train_end": train_end_date_str,
                        "validation_year": validation_year,
                        "look_back": look_back,
                        "case": case_name,
                        "train_months": train_months,
                        "test_months": test_months,
                        "block_idx": "yearly_mean",
                        "yearly_mean_accuracy": mean_acc,
                        "yearly_std_accuracy": std_acc,
                        "num_blocks": int(len(block_accuracies)),
                        "executed_initial_epochs": int(executed_initial_epochs),
                    })

                    print(f"        [RESULTADO] MÉDIA ANUAL {validation_year} ({case_name}): {mean_acc:.4f} ± {std_acc:.4f} ({len(block_accuracies)} blocos)\n")

# =========================
# ao final salvar resultados agregados em CSV
results_df = pd.DataFrame(results)
results_path = os.path.join(os.path.dirname(__file__), "results.csv")
results_df.to_csv(results_path, index=False)

yearly_results = [r for r in results if r.get("block_idx") == "yearly_mean"]
if yearly_results:
    yearly_df = pd.DataFrame(yearly_results)
    yearly_path = os.path.join(os.path.dirname(__file__), "results_yearly_summary.csv")
    yearly_df.to_csv(yearly_path, index=False)
    print(f"\nResumo anual salvo em: {yearly_path}")

print(f"\nTodos os testes concluídos. Resultados salvos em: {results_path}")
