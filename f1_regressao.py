import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Função para converter tempo no formato M:SS.mmm para segundos

def time_to_seconds(t):
    try:
        m, s = t.split(':')
        return int(m) * 60 + float(s)
    except:
        return np.nan

# Função auxiliar para regressão linear simples e plotagem
def run_regression(x, y, x_label, y_label, title):
    mask = ~(np.isnan(x) | np.isnan(y))
    X = x[mask].values.reshape(-1, 1)
    y_vals = y[mask].values
    model = LinearRegression()
    model.fit(X, y_vals)
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_line)
    plt.figure()
    plt.scatter(X, y_vals, alpha=0.5, s=10)
    plt.plot(x_line, y_pred, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def main():
    # Carregar dados
    results = pd.read_csv('archive/results.csv')
    races = pd.read_csv('archive/races.csv')[['raceId', 'year']]
    results = results.merge(races, on='raceId', how='left')

    # Converter colunas numéricas que estão como texto
    to_numeric = ['number', 'position', 'milliseconds', 'fastestLap', 'rank',
                  'fastestLapSpeed']
    for col in to_numeric:
        results[col] = pd.to_numeric(results[col], errors='coerce')
    results['fastestLapTimeSeconds'] = results['fastestLapTime'].apply(time_to_seconds)

    # === Questão 1: Grid vs Pontos ===
    run_regression(results['grid'], results['points'], 'Grid', 'Pontos', 'Grid x Pontos')

    # === Questão 2: Ordem Final vs Pontos ===
    run_regression(results['positionOrder'], results['points'], 'Ordem Final', 'Pontos',
                   'Posição Final x Pontos')

    # === Questão 3: Grid vs Ordem Final ===
    run_regression(results['grid'], results['positionOrder'], 'Grid', 'Ordem Final',
                   'Grid x Ordem Final')

    # === Questão 4: Velocidade da Volta Rápida vs Pontos ===
    run_regression(results['fastestLapSpeed'], results['points'],
                   'Velocidade Volta Rápida', 'Pontos',
                   'Velocidade Volta Rápida x Pontos')

    # === Questão 5: Média de pontos por ano ===
    media_ano = results.groupby('year')['points'].mean().reset_index()
    run_regression(media_ano['year'], media_ano['points'], 'Ano', 'Média de Pontos',
                   'Ano x Média de Pontos')

    # === Questão 6: Voltas vs Tempo Total em Milissegundos ===
    results['milliseconds'] = pd.to_numeric(results['milliseconds'], errors='coerce')
    run_regression(results['laps'], results['milliseconds'], 'Voltas', 'Milissegundos',
                   'Voltas x Milissegundos')

    # === Questão 7: Pontos totais por piloto ===
    pontos_piloto = results.groupby('driverId')['points'].sum().reset_index()
    run_regression(pontos_piloto['driverId'], pontos_piloto['points'], 'ID do Piloto',
                   'Pontos Totais', 'Piloto x Pontos Totais')

    # === Questão 8: Pontos totais por construtor ===
    pontos_constr = results.groupby('constructorId')['points'].sum().reset_index()
    run_regression(pontos_constr['constructorId'], pontos_constr['points'],
                   'ID do Construtor', 'Pontos Totais', 'Construtor x Pontos Totais')

    # === Questão 9: Tempo da Volta Rápida vs Velocidade ===
    run_regression(results['fastestLapTimeSeconds'], results['fastestLapSpeed'],
                   'Tempo Volta Rápida (s)', 'Velocidade',
                   'Tempo da Volta Rápida x Velocidade')

    # === Questão 10: Volta da Volta Rápida vs Pontos ===
    run_regression(results['fastestLap'], results['points'],
                   'Volta da Volta Rápida', 'Pontos',
                   'Volta da Volta Rápida x Pontos')


if __name__ == '__main__':
    main()
