import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Configuração de estilo
sns.set(style="whitegrid")

# Carrega o conjunto de dados de resultados
results = pd.read_csv('archive/results.csv', na_values='\\N')

# Carrega a tabela de corridas para obter o ano de cada prova
races = pd.read_csv('archive/races.csv', na_values='\\N')[['raceId', 'year']]
results = results.merge(races, on='raceId', how='left')

# Converte tempo de volta mais rápida para segundos

def tempo_para_segundos(t):
    if pd.isna(t):
        return np.nan
    minutos, segundos = t.split(':')
    return int(minutos) * 60 + float(segundos)

results['fastestLapTimeSecs'] = results['fastestLapTime'].apply(tempo_para_segundos)

# Função auxiliar para regressão linear simples e plotagem
def regressao_simples(df, x_col, y_col, x_label, y_label, titulo):
    dados = df[[x_col, y_col]].dropna()
    X = dados[[x_col]]
    y = dados[y_col]
    modelo = LinearRegression()
    modelo.fit(X, y)
    intervalo_x = np.linspace(dados[x_col].min(), dados[x_col].max(), 100)
    previsao = modelo.predict(intervalo_x.reshape(-1, 1))
    plt.figure()
    sns.scatterplot(x=x_col, y=y_col, data=dados, alpha=0.5)
    plt.plot(intervalo_x, previsao, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(titulo)
    plt.show()

# === Questao 1 ===
# Relação entre posição no grid e posição final
regressao_simples(results, 'grid', 'positionOrder',
                  'Posição de largada', 'Posição final',
                  'Grid vs. Posição Final')

# === Questao 2 ===
# Posição no grid influenciando na pontuação
regressao_simples(results, 'grid', 'points',
                  'Posição de largada', 'Pontos conquistados',
                  'Grid vs. Pontuação')

# === Questao 3 ===
# Total de voltas completadas e pontos obtidos
regressao_simples(results, 'laps', 'points',
                  'Voltas completadas', 'Pontos',
                  'Voltas Completadas vs. Pontos')

# === Questao 4 ===
# Velocidade da volta mais rápida e pontuação
regressao_simples(results, 'fastestLapSpeed', 'points',
                  'Velocidade da volta mais rápida', 'Pontos',
                  'Velocidade da Volta Rápida vs. Pontos')

# === Questao 5 ===
# Tempo da volta mais rápida (s) e pontuação
regressao_simples(results, 'fastestLapTimeSecs', 'points',
                  'Tempo da volta mais rápida (s)', 'Pontos',
                  'Volta Rápida em segundos vs. Pontos')

# === Questao 6 ===
# Tempo total de prova em milissegundos e pontuação
regressao_simples(results, 'milliseconds', 'points',
                  'Tempo de prova (ms)', 'Pontos',
                  'Tempo de Prova vs. Pontos')

# === Questao 7 ===
# Número do carro e pontuação
regressao_simples(results, 'number', 'points',
                  'Número do carro', 'Pontos',
                  'Número do Carro vs. Pontos')

# === Questao 8 ===
# Posição no grid e tempo total de prova
regressao_simples(results, 'grid', 'milliseconds',
                  'Posição de largada', 'Tempo de prova (ms)',
                  'Grid vs. Tempo de Prova')

# === Questao 9 ===
# Soma de pontos por ano
pontos_ano = results.groupby('year')['points'].sum().reset_index()
regressao_simples(pontos_ano, 'year', 'points',
                  'Ano', 'Soma de pontos',
                  'Soma de Pontos por Ano')

# === Questao 10 ===
# Velocidade média da volta mais rápida por ano
velocidade_ano = results.groupby('year')['fastestLapSpeed'].mean().reset_index()
regressao_simples(velocidade_ano, 'year', 'fastestLapSpeed',
                  'Ano', 'Velocidade média da volta rápida',
                  'Velocidade Média da Volta Rápida por Ano')
