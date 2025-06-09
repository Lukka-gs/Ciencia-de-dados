import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Carrega o dataset de resultados da Fórmula 1
# Colunas com valores ausentes estão marcadas como \N no arquivo
# Usamos pandas para ler o csv localmente

df = pd.read_csv('archive/results.csv')
df.replace('\\N', pd.NA, inplace=True)

# Função auxiliar para preparar colunas numéricas

def prepara_dados(colunas):
    dados = df[colunas].dropna().copy()
    for c in colunas:
        dados[c] = pd.to_numeric(dados[c], errors='coerce')
    dados = dados.dropna()
    return dados

# === Questão 1 ===
# Regressão simples: posição de largada (grid) -> pontos

dados1 = prepara_dados(['grid', 'points'])
X = dados1[['grid']]
y = dados1['points']
modelo1 = LinearRegression()
modelo1.fit(X, y)
prev1 = modelo1.predict(X)
print('Questão 1 - R²:', r2_score(y, prev1))
sns.scatterplot(x='grid', y='points', data=dados1, alpha=0.5)
plt.plot(dados1['grid'], prev1, color='red')
plt.title('Grid vs Points')
plt.show()

# === Questão 2 ===
# Regressão simples: ordem final (positionOrder) -> pontos

dados2 = prepara_dados(['positionOrder', 'points'])
X = dados2[['positionOrder']]
y = dados2['points']
modelo2 = LinearRegression()
modelo2.fit(X, y)
prev2 = modelo2.predict(X)
print('Questão 2 - R²:', r2_score(y, prev2))
sns.scatterplot(x='positionOrder', y='points', data=dados2, alpha=0.5)
plt.plot(dados2['positionOrder'], prev2, color='red')
plt.title('Position Order vs Points')
plt.show()

# === Questão 3 ===
# Regressão simples: voltas completadas (laps) -> pontos

dados3 = prepara_dados(['laps', 'points'])
X = dados3[['laps']]
y = dados3['points']
modelo3 = LinearRegression()
modelo3.fit(X, y)
prev3 = modelo3.predict(X)
print('Questão 3 - R²:', r2_score(y, prev3))
sns.scatterplot(x='laps', y='points', data=dados3, alpha=0.5)
plt.plot(dados3['laps'], prev3, color='red')
plt.title('Laps vs Points')
plt.show()

# === Questão 4 ===
# Regressão simples: identificador da corrida (raceId) -> pontos

dados4 = prepara_dados(['raceId', 'points'])
X = dados4[['raceId']]
y = dados4['points']
modelo4 = LinearRegression()
modelo4.fit(X, y)
prev4 = modelo4.predict(X)
print('Questão 4 - R²:', r2_score(y, prev4))
sns.scatterplot(x='raceId', y='points', data=dados4, alpha=0.5)
plt.plot(dados4['raceId'], prev4, color='red')
plt.title('Race ID vs Points')
plt.show()

# === Questão 5 ===
# Regressão simples: volta mais rápida (fastestLap) -> pontos

dados5 = prepara_dados(['fastestLap', 'points'])
X = dados5[['fastestLap']]
y = dados5['points']
modelo5 = LinearRegression()
modelo5.fit(X, y)
prev5 = modelo5.predict(X)
print('Questão 5 - R²:', r2_score(y, prev5))
sns.scatterplot(x='fastestLap', y='points', data=dados5, alpha=0.5)
plt.plot(dados5['fastestLap'], prev5, color='red')
plt.title('Fastest Lap vs Points')
plt.show()

# === Questão 6 ===
# Regressão simples: status da corrida (statusId) -> pontos

dados6 = prepara_dados(['statusId', 'points'])
X = dados6[['statusId']]
y = dados6['points']
modelo6 = LinearRegression()
modelo6.fit(X, y)
prev6 = modelo6.predict(X)
print('Questão 6 - R²:', r2_score(y, prev6))
sns.scatterplot(x='statusId', y='points', data=dados6, alpha=0.5)
plt.plot(dados6['statusId'], prev6, color='red')
plt.title('Status ID vs Points')
plt.show()

# === Questão 7 ===
# Regressão múltipla: grid, positionOrder e laps -> points
# Dividimos em treino e teste e exibimos R² e erro médio

dados7 = prepara_dados(['grid', 'positionOrder', 'laps', 'points'])
X = dados7[['grid', 'positionOrder', 'laps']]
y = dados7['points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelo7 = LinearRegression()
modelo7.fit(X_train, y_train)
prev7 = modelo7.predict(X_test)
print('Questão 7 - R²:', r2_score(y_test, prev7))
print('Questão 7 - Erro médio:', mean_absolute_error(y_test, prev7))
plt.scatter(y_test, prev7, alpha=0.5)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Predicted vs Actual Points')
plt.show()

# === Questão 8 ===
# Regressão simples: grid -> laps

dados8 = prepara_dados(['grid', 'laps'])
X = dados8[['grid']]
y = dados8['laps']
modelo8 = LinearRegression()
modelo8.fit(X, y)
prev8 = modelo8.predict(X)
print('Questão 8 - R²:', r2_score(y, prev8))
sns.scatterplot(x='grid', y='laps', data=dados8, alpha=0.5)
plt.plot(dados8['grid'], prev8, color='red')
plt.title('Grid vs Laps')
plt.show()

# === Questão 9 ===
# Regressão simples: fastestLap -> fastestLapSpeed

dados9 = df[['fastestLap', 'fastestLapSpeed']].dropna().copy()
dados9['fastestLap'] = pd.to_numeric(dados9['fastestLap'], errors='coerce')
dados9['fastestLapSpeed'] = pd.to_numeric(dados9['fastestLapSpeed'], errors='coerce')
dados9 = dados9.dropna()
X = dados9[['fastestLap']]
y = dados9['fastestLapSpeed']
modelo9 = LinearRegression()
modelo9.fit(X, y)
prev9 = modelo9.predict(X)
print('Questão 9 - R²:', r2_score(y, prev9))
sns.scatterplot(x='fastestLap', y='fastestLapSpeed', data=dados9, alpha=0.5)
plt.plot(dados9['fastestLap'], prev9, color='red')
plt.title('Fastest Lap vs Speed')
plt.show()

# === Questão 10 ===
# Regressão simples: constructorId -> points

dados10 = prepara_dados(['constructorId', 'points'])
X = dados10[['constructorId']]
y = dados10['points']
modelo10 = LinearRegression()
modelo10.fit(X, y)
prev10 = modelo10.predict(X)
print('Questão 10 - R²:', r2_score(y, prev10))
sns.scatterplot(x='constructorId', y='points', data=dados10, alpha=0.5)
plt.plot(dados10['constructorId'], prev10, color='red')
plt.title('Constructor ID vs Points')
plt.show()
