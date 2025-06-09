import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Carrega o dataset results.csv localizado na pasta archive
# Converte colunas que deveriam ser numéricas

df = pd.read_csv('archive/results.csv')
num_cols = [
    'number', 'grid', 'positionOrder', 'points', 'laps',
    'milliseconds', 'fastestLap', 'rank',
    'fastestLapSpeed', 'statusId', 'raceId',
    'driverId', 'constructorId'
]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


def regressao_simples(data, x_col, y_col):
    """Realiza regressão linear simples entre duas colunas."""
    dados = data[[x_col, y_col]].dropna()
    X = dados[[x_col]]
    y = dados[y_col]
    modelo = LinearRegression()
    modelo.fit(X, y)
    pred = modelo.predict(X)
    r2 = r2_score(y, pred)

    sns.scatterplot(x=x_col, y=y_col, data=dados, s=10)
    sns.lineplot(x=dados[x_col], y=pred, color='red')
    plt.title(f'Regressão {y_col} ~ {x_col}\nR² = {r2:.4f}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()


# === Questão 1 ===
# Relação entre grid (posição de largada) e positionOrder (posição final)
regressao_simples(df, 'grid', 'positionOrder')

# === Questão 2 ===
# Regressão entre positionOrder e points
regressao_simples(df, 'positionOrder', 'points')

# === Questão 3 ===
# Relação entre grid e points
regressao_simples(df, 'grid', 'points')

# === Questão 4 ===
# Relação entre laps completadas e points
regressao_simples(df, 'laps', 'points')

# === Questão 5 ===
# Relação entre tempo total em milliseconds e points
regressao_simples(df, 'milliseconds', 'points')

# === Questão 6 ===
# Relação entre velocidade da volta mais rápida e points
regressao_simples(df, 'fastestLapSpeed', 'points')

# === Questão 7 ===
# Relação entre número da volta mais rápida e positionOrder
regressao_simples(df, 'fastestLap', 'positionOrder')

# === Questão 8 ===
# Relação entre rank da volta mais rápida e positionOrder
regressao_simples(df, 'rank', 'positionOrder')

# === Questão 9 ===
# Relação entre número do carro e points
regressao_simples(df, 'number', 'points')

# === Questão 10 ===
# Relação entre statusId e positionOrder
regressao_simples(df, 'statusId', 'positionOrder')
