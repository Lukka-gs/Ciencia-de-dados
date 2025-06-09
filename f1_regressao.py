# -*- coding: utf-8 -*-
"""Analises de regressao linear usando dados de Formula 1."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Carrega dados
results = pd.read_csv('archive/results.csv')
results = results.replace('\\N', pd.NA)

# Conversoes numericas
for col in ['number', 'position', 'positionText', 'milliseconds', 'fastestLap', 'rank', 'fastestLapSpeed']:
    results[col] = pd.to_numeric(results[col], errors='coerce')

def time_to_seconds(x):
    if pd.isna(x):
        return np.nan
    try:
        m, s = x.split(':')
        return int(m) * 60 + float(s)
    except ValueError:
        return np.nan

results['fastestLapTime'] = results['fastestLapTime'].apply(time_to_seconds)

# Funcao auxiliar para regressao linear

def simple_linear_regression(x, y, x_label, y_label, title):
    df = results[[x, y]].dropna()
    X = df[[x]].values
    y_vals = df[y].values
    model = LinearRegression()
    model.fit(X, y_vals)
    pred = model.predict(X)
    r2 = r2_score(y_vals, pred)

    sns.scatterplot(x=x, y=y, data=df, color='blue', alpha=0.5)
    sns.lineplot(x=df[x], y=pred, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}\nR²={r2:.4f}")
    plt.show()
    plt.close()

# === Questao 1 ===
# Relacao entre posicao de largada (grid) e pontos conquistados
simple_linear_regression('grid', 'points', 'Grid', 'Pontos', 'Grid vs Pontos')

# === Questao 2 ===
# Pontos em funcao da ordem final de chegada
simple_linear_regression('positionOrder', 'points', 'Posicao final', 'Pontos', 'Posicao final vs Pontos')

# === Questao 3 ===
# Voltas completadas em funcao da posicao de largada
simple_linear_regression('grid', 'laps', 'Grid', 'Voltas', 'Grid vs Voltas completadas')

# === Questao 4 ===
# Voltas completadas e pontos
simple_linear_regression('laps', 'points', 'Voltas', 'Pontos', 'Voltas vs Pontos')

# === Questao 5 ===
# Numero da volta mais rapida e sua velocidade
simple_linear_regression('fastestLap', 'fastestLapSpeed', 'Volta mais rapida', 'Velocidade (km/h)', 'Volta mais rapida vs Velocidade')

# === Questao 6 ===
# Regressao linear entre posicao de largada (grid) e ordem de chegada (positionOrder).
# Analisa se largar na frente influencia a posicao final na corrida.
simple_linear_regression('grid', 'positionOrder', 'Grid', 'Ordem de chegada', 'Grid vs Ordem de chegada')

# === Questao 7 ===
# Tempo de volta mais rapida e velocidade
simple_linear_regression('fastestLapTime', 'fastestLapSpeed', 'Tempo da volta (s)', 'Velocidade (km/h)', 'Tempo da volta vs Velocidade')

# === Questao 8 ===
# Colocacao na volta mais rapida (rank) em funcao do numero da volta
simple_linear_regression('fastestLap', 'rank', 'Volta mais rapida', 'Rank', 'Volta mais rapida vs Rank')

# === Questao 9 ===
# Relacao entre identificacao da corrida (raceId) e pontos obtidos
simple_linear_regression('raceId', 'points', 'ID da corrida', 'Pontos', 'ID da corrida vs Pontos')

# === Questao 10 ===
# Numero do carro e ordem final de chegada
simple_linear_regression('number', 'positionOrder', 'Numero do carro', 'Ordem de chegada', 'Numero do carro vs Ordem de chegada')
