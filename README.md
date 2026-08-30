# Previsão de tendências com LSTM

Este repositório contém dois scripts Python para classificar a tendência do preço de fechamento de ativos financeiros em três categorias: **queda**, **estabilidade** ou **alta**. Os scripts utilizam redes neurais recorrentes do tipo **LSTM**, dados armazenados em MongoDB e avaliação em períodos temporais distintos.

Os códigos têm objetivos complementares:

- `lstm_anos.py`: avalia períodos históricos de 3 e 5 anos, com divisão treino/teste e retreinamento incremental durante o teste.
- `lstm_meses.py`: treina com um período anterior e realiza avaliação deslizante em blocos dentro de um ano de validação.

## 1. `lstm_anos.py`

O script percorre os ativos definidos em `stock_tickers` — atualmente `PSSA3` e `BRFS3` —, os anos `2022`, `2023` e `2024`, e os modos de período `5y` e `3y`.

Para cada combinação, os dados são consultados na coleção `stock.papers` do MongoDB. No modo `5y`, busca-se um intervalo iniciado cinco anos antes do ano avaliado e encerrado em 31 de dezembro desse ano. No modo `3y`, busca-se um intervalo iniciado três anos antes. Para os anos de 2022 e 2023, o intervalo pode avançar até 30 de junho do ano seguinte; para 2024, termina em 31 de dezembro de 2024.

No modo `5y`, as sequências são divididas em aproximadamente 80% para treinamento e 20% para teste. No modo `3y`, o treinamento é feito até o final do ano anterior, e o teste é dividido em dois subperíodos de aproximadamente nove meses.

O script testa os valores de `look_back` definidos em `look_backs`: `1`, `3`, `5`, `10`, `20`, `30`, `60` e `90`.

## 2. `lstm_meses.py`

O script está configurado para processar o ativo `WEGE3` e o ano de validação `2021`. O caso ativo em `sliding_cases` é `3m`, que utiliza os três meses anteriores ao ano de validação para treinamento e um bloco de aproximadamente um mês para cada etapa de teste. Os casos de 6 meses e 1 ano estão presentes no código, mas permanecem comentados.

O treinamento usa dados do período de outubro a dezembro do ano anterior. A validação usa os dados de janeiro a dezembro do ano configurado em `validation_years`.

Durante a validação, o ano é percorrido em blocos de aproximadamente `30,44` dias. Para cada bloco, o script seleciona uma janela de sequências anteriores, ajusta um novo modelo LSTM e calcula a acurácia do bloco. Ao final, calcula a média e o desvio padrão das acurácias dos blocos.

Os valores testados de `look_back` são `1`, `3`, `5`, `10`, `20` e `30`.

## 3. Lógica do algoritmo

### 3.1 Preparação dos dados

Os dois scripts consultam o MongoDB local usando:

- banco: `stock`;
- coleção: `papers`;
- filtro de ativo: campo `codneg`;
- filtro temporal: campo `date`;
- preço utilizado: campo `preult`.

O campo `preult` é renomeado para `Close`. Os registros são convertidos para datas, ordenados cronologicamente e transformados em rótulos de tendência:

- `0`: queda — o próximo `Close` é menor que o atual;
- `1`: estabilidade — o próximo `Close` é igual ao atual;
- `2`: alta — o próximo `Close` é maior que o atual.

O último registro é descartado porque não possui um preço seguinte para comparação.

### 3.2 Criação das sequências

A função `create_sequences(data, labels, look_back)` cria amostras supervisionadas com uma janela temporal:

- `X`: sequência dos últimos `look_back` preços;
- `y`: classe de tendência associada ao preço seguinte à janela.

Assim, o modelo recebe uma sequência de preços de fechamento e retorna uma das três classes de tendência.

### 3.3 Escalonamento

Antes do treinamento, os preços são normalizados com `MinMaxScaler(feature_range=(0, 1))`. O escalonador é ajustado nos dados de treinamento e aplicado às sequências correspondentes. No `lstm_meses.py`, cada bloco deslizante possui um escalonador próprio, ajustado somente no conjunto de treinamento daquele bloco.

### 3.4 Arquitetura LSTM

A função `build_lstm_model(input_shape)` constrói o mesmo modelo nos dois scripts:

1. uma camada `LSTM` com 100 unidades e `return_sequences=True`;
2. uma camada `Dropout(0.2)`;
3. uma segunda camada `LSTM` com 100 unidades;
4. uma segunda camada `Dropout(0.2)`;
5. uma camada de saída `Dense(3, activation='softmax')`.

O modelo é compilado com o otimizador `adam`, a função de perda `categorical_crossentropy` e a métrica `accuracy`.

Os rótulos são convertidos para representação one-hot por meio de `to_categorical`. Para reduzir o efeito de classes desbalanceadas, `make_class_weight` calcula pesos com `compute_class_weight` quando possível.

### 3.5 Determinismo e treinamento

Os scripts definem a semente `SEED = 12345` para Python, NumPy e TensorFlow. Também configuram variáveis de ambiente e limitam o paralelismo do TensorFlow para favorecer resultados reproduzíveis. A função `tf.config.experimental.enable_op_determinism()` é ativada quando disponível.

O treinamento inicial utiliza até `100` épocas, lote de `32` amostras e `shuffle=False`. Quando há mais de 10 sequências, é utilizado `validation_split=0.1` e `EarlyStopping` com paciência de `15` épocas. `ModelCheckpoint` salva os melhores pesos na pasta `checkpoints`.

No `lstm_anos.py`, após cada previsão do conjunto de teste, o exemplo avaliado e seu rótulo real são adicionados a um `deque`. Esse buffer é usado para o retreinamento incremental do modelo antes da próxima previsão. O tamanho do buffer é definido por `window_size`; quando seu valor é `None`, utiliza-se todo o conjunto de treinamento.

No `lstm_meses.py`, cada bloco de validação cria e treina um novo modelo (`model_slide`) usando as sequências anteriores disponíveis, respeitando o tamanho real do conjunto de treinamento inicial.

## 4. Métricas e arquivos gerados

O `lstm_anos.py` calcula:

- `accuracy`: proporção de classes previstas corretamente;
- `rmse`: raiz do erro quadrático médio entre os códigos numéricos previstos e reais;
- no modo `3y`, média e desvio padrão das acurácias dos dois subperíodos.

O `lstm_meses.py` calcula:

- acurácia de cada bloco de validação;
- média anual das acurácias dos blocos;
- desvio padrão anual das acurácias;
- quantidade de blocos e épocas executadas.

Os dois scripts salvam `results.csv`. O `lstm_meses.py` também salva `results_yearly_summary.csv` quando existem resultados anuais agregados. Os checkpoints são salvos em `checkpoints/`.

## 5. Requisitos e instalação

É necessário utilizar Python e instalar as bibliotecas importadas pelos scripts:

```bash
pip install numpy pandas pymongo scikit-learn tensorflow
```

Também é necessário ter um servidor MongoDB acessível em:

```text
mongodb://localhost:27017/
```

A base `stock` deve conter a coleção `papers` com, no mínimo, os campos:

- `codneg`: código do ativo;
- `date`: data do registro;
- `preult`: preço de fechamento utilizado pelo algoritmo.

O campo `date` deve ser compatível com a conversão realizada por `pandas.to_datetime`, e os registros devem conter valores válidos em `preult`.

## 6. Como executar

Com o MongoDB em execução e os arquivos no diretório do projeto:

```bash
python lstm_anos.py
```

ou:

```bash
python lstm_meses.py
```

Os ativos, anos, períodos, valores de `look_back`, número de épocas e demais parâmetros podem ser alterados diretamente nas variáveis de configuração de cada script.

## 7. Limitações e cuidados

- A conexão com o MongoDB está fixa em `mongodb://localhost:27017/`; altere a URI se o servidor estiver em outro endereço.
- Os ativos, anos e períodos analisados estão definidos diretamente no código.
- O modelo classifica tendências a partir da comparação do preço atual com o próximo preço; ele não prevê diretamente um valor futuro de preço.
- A classe de estabilidade só ocorre quando os dois preços comparados são exatamente iguais.
- O `rmse` é calculado sobre os códigos das classes (`0`, `1` e `2`), portanto deve ser interpretado como uma medida auxiliar de erro entre classes ordenadas, não como erro monetário.
- O `lstm_anos.py` utiliza o rótulo real de cada observação para atualizar o modelo durante o teste. Esse procedimento representa um cenário de atualização incremental com informação observada após cada passo e deve ser considerado ao comparar resultados.
- O `lstm_meses.py` está configurado para um ativo, um ano e o caso `3m`; outros cenários precisam ser habilitados ou ajustados no código.
- Não há tratamento explícito para valores ausentes, outliers, falhas de conexão ou validação dos tipos dos campos do MongoDB.
- Os scripts podem gerar muitos treinamentos, pois repetem o processo para vários valores de `look_back`, ativos, períodos e blocos temporais.
