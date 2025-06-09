import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Carregar dados tratando valores ausentes representados por \N
arquivo = 'archive/results.csv'
df = pd.read_csv(arquivo, na_values='\\N')

# Converter fastestLapTime (min:seg.milis) para segundos
if 'fastestLapTime' in df.columns:
    def tempo_para_segundos(txt):
        try:
            minutos, segundos = txt.split(':')
            return int(minutos) * 60 + float(segundos)
        except Exception:
            return None
    df['fastestLapTime_sec'] = df['fastestLapTime'].apply(tempo_para_segundos)

# Função auxiliar para regressão e gráfico

def regressao_simples(data, x_col, y_col, titulo):
    dados = data[[x_col, y_col]].dropna()
    if dados.empty:
        print(f'Sem dados suficientes para {titulo}')
        return
    X = dados[[x_col]]
    y = dados[y_col]
    modelo = LinearRegression()
    modelo.fit(X, y)
    pred = modelo.predict(X)
    r2 = r2_score(y, pred)
    print(f'{titulo}\nCoeficiente: {modelo.coef_[0]:.4f} Intercept: {modelo.intercept_:.4f} R2: {r2:.4f}\n')
    sns.scatterplot(x=x_col, y=y_col, data=dados, alpha=0.5)
    plt.plot(dados[x_col], pred, color='red')
    plt.title(titulo)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

# === Questão 1 ===
regressao_simples(df, 'grid', 'positionOrder', 'Grid vs Posicao Final')

# === Questão 2 ===
regressao_simples(df, 'grid', 'points', 'Grid vs Pontos')

# === Questão 3 ===
regressao_simples(df, 'laps', 'points', 'Laps vs Pontos')

# === Questão 4 ===
regressao_simples(df, 'positionOrder', 'points', 'Posicao Final vs Pontos')

# === Questão 5 ===
regressao_simples(df, 'milliseconds', 'points', 'Tempo em ms vs Pontos')

# === Questão 6 ===
regressao_simples(df, 'fastestLapSpeed', 'points', 'Velocidade da Volta Mais Rápida vs Pontos')

# === Questão 7 ===
regressao_simples(df, 'rank', 'fastestLapSpeed', 'Rank da Volta Mais Rápida vs Velocidade')

# === Questão 8: fastestLapTime vs points ===
if 'fastestLapTime_sec' in df.columns:
    regressao_simples(df, 'fastestLapTime_sec', 'points', 'fastestLapTime vs Pontos')
else:
    print('Coluna fastestLapTime não encontrada. Pulando Questão 8.')

# === Questão 9 ===
regressao_simples(df, 'number', 'points', 'Numero do Carro vs Pontos')

# === Questão 10 ===
regressao_simples(df, 'raceId', 'points', 'ID da Corrida vs Pontos')
