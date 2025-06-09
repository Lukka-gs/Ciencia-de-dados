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
=======
from sklearn.model_selection import train_test_split

# Carregar dados
arquivo = 'archive/results.csv'
df = pd.read_csv(arquivo)

# === Limpeza Inicial ===
# Substitui '\\N' por NA e converte colunas numéricas

# \N é usado como marcador de valores ausentes nesse dataset

df.replace('\\N', pd.NA, inplace=True)
for col in ['milliseconds', 'fastestLapSpeed']:
    df[col] = pd.to_numeric(df[col])

# Converter tempo de volta para segundos

def tempo_para_segundos(txt):
    """Converte string M:SS.mmm em segundos."""
    if pd.isna(txt):
        return None
    minutos, segundos = txt.split(':')
    return int(minutos) * 60 + float(segundos)

df['fastestLapTime_sec'] = df['fastestLapTime'].apply(tempo_para_segundos)

# === Questao 1 ===
# Relacao entre posicao de largada (grid) e pontos
q1 = df[['grid', 'points']].dropna()
X1 = q1[['grid']]
y1 = q1['points']
modelo_q1 = LinearRegression().fit(X1, y1)
print('Questao 1 - coeficiente:', modelo_q1.coef_[0], 'intercepto:', modelo_q1.intercept_)
sns.regplot(x='grid', y='points', data=q1, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Grid vs Pontos')
plt.xlabel('Posicao de largada')
plt.ylabel('Pontos conquistados')
plt.show()

# === Questao 2 ===
# Pontos em funcao da posicao final (positionOrder)
q2 = df[['positionOrder', 'points']].dropna()
X2 = q2[['positionOrder']]
y2 = q2['points']
modelo_q2 = LinearRegression().fit(X2, y2)
print('Questao 2 - coeficiente:', modelo_q2.coef_[0], 'intercepto:', modelo_q2.intercept_)
sns.regplot(x='positionOrder', y='points', data=q2, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Posicao Final vs Pontos')
plt.xlabel('Posicao final')
plt.ylabel('Pontos conquistados')
plt.show()

# === Questao 3 ===
# Pontos em funcao de voltas completadas
q3 = df[['laps', 'points']].dropna()
X3 = q3[['laps']]
y3 = q3['points']
modelo_q3 = LinearRegression().fit(X3, y3)
print('Questao 3 - coeficiente:', modelo_q3.coef_[0], 'intercepto:', modelo_q3.intercept_)
sns.regplot(x='laps', y='points', data=q3, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Voltas vs Pontos')
plt.xlabel('Voltas completadas')
plt.ylabel('Pontos conquistados')
plt.show()

# === Questao 4 ===
# Posicao final em funcao do grid de largada
q4 = df[['grid', 'positionOrder']].dropna()
X4 = q4[['grid']]
y4 = q4['positionOrder']
modelo_q4 = LinearRegression().fit(X4, y4)
print('Questao 4 - coeficiente:', modelo_q4.coef_[0], 'intercepto:', modelo_q4.intercept_)
sns.regplot(x='grid', y='positionOrder', data=q4, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Grid vs Posicao Final')
plt.xlabel('Posicao de largada')
plt.ylabel('Posicao final')
plt.show()

# === Questao 5 ===
# Velocidade da volta mais rapida em funcao do tempo de volta
q5 = df[['fastestLapTime_sec', 'fastestLapSpeed']].dropna()
X5 = q5[['fastestLapTime_sec']]
y5 = q5['fastestLapSpeed']
modelo_q5 = LinearRegression().fit(X5, y5)
print('Questao 5 - coeficiente:', modelo_q5.coef_[0], 'intercepto:', modelo_q5.intercept_)
sns.regplot(x='fastestLapTime_sec', y='fastestLapSpeed', data=q5, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Tempo da Volta x Velocidade')
plt.xlabel('Tempo da volta (s)')
plt.ylabel('Velocidade (km/h)')
plt.show()

# === Questao 6 ===
# Pontos em funcao da velocidade da volta mais rapida
q6 = df[['fastestLapSpeed', 'points']].dropna()
X6 = q6[['fastestLapSpeed']]
y6 = q6['points']
modelo_q6 = LinearRegression().fit(X6, y6)
print('Questao 6 - coeficiente:', modelo_q6.coef_[0], 'intercepto:', modelo_q6.intercept_)
sns.regplot(x='fastestLapSpeed', y='points', data=q6, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Velocidade da Volta Rapida vs Pontos')
plt.xlabel('Velocidade (km/h)')
plt.ylabel('Pontos')
plt.show()

# === Questao 7 ===
# Regressao multipla: grid, positionOrder e laps explicando pontos
q7 = df[['grid', 'positionOrder', 'laps', 'points']].dropna()
X7 = q7[['grid', 'positionOrder', 'laps']]
y7 = q7['points']
modelo_q7 = LinearRegression().fit(X7, y7)
print('Questao 7 - coeficientes:', modelo_q7.coef_, 'intercepto:', modelo_q7.intercept_)
print('Exemplos de predicao:', modelo_q7.predict(X7.head()))

# === Questao 8 ===
# Avaliacao do modelo da questao 7 com separacao treino/teste
X_train, X_test, y_train, y_test = train_test_split(X7, y7, test_size=0.3, random_state=42)
modelo_q8 = LinearRegression().fit(X_train, y_train)
print('Questao 8 - R^2 de teste:', modelo_q8.score(X_test, y_test))

# === Questao 9 ===
# Tempo total em funcao do numero de voltas
q9 = df[['laps', 'milliseconds']].dropna()
X9 = q9[['laps']]
y9 = q9['milliseconds']
modelo_q9 = LinearRegression().fit(X9, y9)
print('Questao 9 - coeficiente:', modelo_q9.coef_[0], 'intercepto:', modelo_q9.intercept_)
sns.regplot(x='laps', y='milliseconds', data=q9, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Voltas vs Tempo Total (ms)')
plt.xlabel('Voltas completadas')
plt.ylabel('Tempo total (ms)')
plt.show()

# === Questao 10: Funcao prever_pontos(grid, posicao_final, voltas) ===
def prever_pontos(grid, posicao_final, voltas):
    """Prevê pontos usando o modelo da questao 7."""
    valores = [[grid, posicao_final, voltas]]
    return modelo_q7.predict(valores)[0]

# Testes da funcao
exemplos = [
    (1, 1, 58),
    (10, 15, 56),
    (5, 4, 60)
]
for g, p, v in exemplos:
    resultado = prever_pontos(g, p, v)
    print(f'Previsao de pontos para grid={g}, posicao_final={p}, voltas={v}: {resultado:.2f}')
