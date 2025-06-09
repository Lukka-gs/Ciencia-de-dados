import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Questao 1: Carregamento e visualizacao inicial ===
# Leitura do arquivo de resultados e exibicao das primeiras linhas e estatisticas

df = pd.read_csv('archive/results.csv')

print(df.head())
print(df.describe())

# === Questao 2: Grid x Position Order ===
# Relacao entre a posicao de largada e a ordem final de chegada

data = df[['grid', 'positionOrder']].apply(pd.to_numeric, errors='coerce').dropna()
X = data[['grid']]
y = data['positionOrder']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q2 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['grid'], y=y, alpha=0.3)
plt.plot(data['grid'], pred, color='red')
plt.xlabel('Grid')
plt.ylabel('Posicao de Chegada')
plt.title('Grid vs Position Order')
plt.show()

# === Questao 3: Grid x Pontos ===
# Analisa se a posicao de largada interfere nos pontos obtidos

data = df[['grid', 'points']].apply(pd.to_numeric, errors='coerce').dropna()
X = data[['grid']]
y = data['points']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q3 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['grid'], y=y, alpha=0.3)
plt.plot(data['grid'], pred, color='red')
plt.xlabel('Grid')
plt.ylabel('Pontos')
plt.title('Grid vs Pontos')
plt.show()

# === Questao 4: Laps x Pontos ===
# Investigacao do numero de voltas completadas em relacao aos pontos

data = df[['laps', 'points']].apply(pd.to_numeric, errors='coerce').dropna()
X = data[['laps']]
y = data['points']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q4 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['laps'], y=y, alpha=0.3)
plt.plot(data['laps'], pred, color='red')
plt.xlabel('Voltas')
plt.ylabel('Pontos')
plt.title('Voltas vs Pontos')
plt.show()

# === Questao 5: Velocidade da Volta Mais Rapida x Pontos ===
# Usa a velocidade da volta mais rapida para prever pontuacao

data = df[['fastestLapSpeed', 'points']].replace('\\N', pd.NA).dropna()
data['fastestLapSpeed'] = pd.to_numeric(data['fastestLapSpeed'])
X = data[['fastestLapSpeed']]
y = data['points']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q5 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['fastestLapSpeed'], y=y, alpha=0.3)
plt.plot(data['fastestLapSpeed'], pred, color='red')
plt.xlabel('Velocidade da Volta Mais Rapida')
plt.ylabel('Pontos')
plt.title('Velocidade da Volta Mais Rapida vs Pontos')
plt.show()

# === Questao 6: Velocidade da Volta Mais Rapida x Position Order ===
# Relacao entre velocidade da melhor volta e posicao final

data = df[['fastestLapSpeed', 'positionOrder']].replace('\\N', pd.NA).dropna()
data['fastestLapSpeed'] = pd.to_numeric(data['fastestLapSpeed'])
X = data[['fastestLapSpeed']]
y = data['positionOrder']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q6 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['fastestLapSpeed'], y=y, alpha=0.3)
plt.plot(data['fastestLapSpeed'], pred, color='red')
plt.xlabel('Velocidade da Volta Mais Rapida')
plt.ylabel('Posicao de Chegada')
plt.title('Velocidade da Volta Mais Rapida vs Posicao')
plt.show()

# === Questao 7: Tempo de Prova (ms) x Pontos ===
# Relacao entre o tempo total de prova e os pontos conquistados

data = df[['milliseconds', 'points']].replace('\\N', pd.NA).dropna()
data['milliseconds'] = pd.to_numeric(data['milliseconds'])
X = data[['milliseconds']]
y = data['points']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q7 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['milliseconds'], y=y, alpha=0.3)
plt.plot(data['milliseconds'], pred, color='red')
plt.xlabel('Tempo em ms')
plt.ylabel('Pontos')
plt.title('Tempo de Prova (ms) vs Pontos')
plt.show()

# === Questao 8: Grid x Tempo de Prova (ms) ===
# Verifica se a posicao de largada influencia no tempo total de prova

data = df[['grid', 'milliseconds']].replace('\\N', pd.NA).dropna()
data['milliseconds'] = pd.to_numeric(data['milliseconds'])
X = data[['grid']]
y = data['milliseconds']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q8 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['grid'], y=y, alpha=0.3)
plt.plot(data['grid'], pred, color='red')
plt.xlabel('Grid')
plt.ylabel('Tempo em ms')
plt.title('Grid vs Tempo de Prova')
plt.show()

# === Questao 9: Construtor x Pontos ===
# Avalia a relacao entre a equipe (constructorId) e os pontos marcados

data = df[['constructorId', 'points']].apply(pd.to_numeric, errors='coerce').dropna()
X = data[['constructorId']]
y = data['points']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q9 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['constructorId'], y=y, alpha=0.3)
plt.plot(data['constructorId'], pred, color='red')
plt.xlabel('ID do Construtor')
plt.ylabel('Pontos')
plt.title('Construtor vs Pontos')
plt.show()

# === Questao 10: Piloto x Pontos ===
# Analise simples entre o piloto (driverId) e os pontos obtidos

data = df[['driverId', 'points']].apply(pd.to_numeric, errors='coerce').dropna()
X = data[['driverId']]
y = data['points']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Q10 - Coeficiente:', model.coef_[0], 'Intercepto:', model.intercept_, 'R2:', r2_score(y, pred))

sns.scatterplot(x=data['driverId'], y=y, alpha=0.3)
plt.plot(data['driverId'], pred, color='red')
plt.xlabel('ID do Piloto')
plt.ylabel('Pontos')
plt.title('Piloto vs Pontos')
plt.show()
