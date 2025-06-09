import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Carrega o dataset
# Substitui marcadores de ausência '\\N' por NA do pandas

df = pd.read_csv('archive/results.csv')
df.replace('\\N', pd.NA, inplace=True)

# Converte colunas numéricas
numeric_cols = ['grid','positionOrder','points','laps','milliseconds','fastestLap',
                'rank','fastestLapSpeed','driverId','raceId','constructorId','number',
                'position']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Função para converter tempos (ex.: '1:27.452') para segundos

def parse_time(value: str):
    if pd.isna(value):
        return np.nan
    try:
        parts = value.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(value)
    except Exception:
        return np.nan

# Aplica conversão de tempo

df['fastestLapTimeSec'] = df['fastestLapTime'].apply(parse_time)
df['timeSec'] = df['time'].apply(parse_time)

# Função auxiliar para regressão linear simples e plotagem
def simple_regression(dataframe, x_col, y_col, titulo):
    data = dataframe[[x_col, y_col]].dropna()
    X = data[[x_col]]
    y = data[y_col]

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    corr = data[x_col].corr(data[y_col])

    # Resultados numéricos
    print(titulo)
    print(f'Coeficiente: {model.coef_[0]:.4f}')
    print(f'Intercepto: {model.intercept_:.4f}')
    print(f'R^2: {r2:.4f}')
    print(f'Correlacao: {corr:.4f}\n')

    # Gráfico
    sns.scatterplot(x=x_col, y=y_col, data=data, alpha=0.5)
    plt.plot(data[x_col], model.predict(X), color='red')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(titulo)
    plt.show()

# === Questão 1 ===
# Grid de largada versus pontos
simple_regression(df, 'grid', 'points', 'Grid vs Points')

# === Questão 2 ===
# Ordem de chegada versus pontos
simple_regression(df, 'positionOrder', 'points', 'Position Order vs Points')

# === Questão 3 ===
# Velocidade da volta mais rápida versus pontos
simple_regression(df, 'fastestLapSpeed', 'points', 'Fastest Lap Speed vs Points')

# === Questão 4 ===
# Voltass completadas versus pontos
simple_regression(df, 'laps', 'points', 'Laps vs Points')

# === Questão 5 ===
# Tempo da volta mais rápida em segundos versus pontos
simple_regression(df, 'fastestLapTimeSec', 'points', 'Fastest Lap Time vs Points')

# === Questão 6 ===
# Grid de largada versus ordem de chegada
simple_regression(df, 'grid', 'positionOrder', 'Grid vs Position Order')

# === Questão 7 ===
# Voltass completadas versus tempo em milissegundos
simple_regression(df, 'laps', 'milliseconds', 'Laps vs Milliseconds')

# === Questão 8 ===
# Colocação da volta mais rápida versus sua velocidade
simple_regression(df, 'rank', 'fastestLapSpeed', 'Rank vs Fastest Lap Speed')

# === Questão 9 ===
# Grid de largada versus tempo em milissegundos
simple_regression(df, 'grid', 'milliseconds', 'Grid vs Milliseconds')

# === Questão 10 ===
# Identificação do piloto versus pontos
simple_regression(df, 'driverId', 'points', 'Driver ID vs Points')
