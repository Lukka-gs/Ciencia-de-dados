import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Carrega os dados
results = pd.read_csv('archive/results.csv')

# Converte colunas importantes para numérico
numeric_cols = ['grid', 'positionOrder', 'points', 'laps', 'milliseconds',
                'fastestLap', 'rank', 'fastestLapSpeed']
for col in numeric_cols:
    results[col] = pd.to_numeric(results[col], errors='coerce')

# Remove linhas com valores ausentes nas colunas numéricas
results.dropna(subset=numeric_cols, inplace=True)

# Função auxiliar para ajustar e exibir regressão linear

def linear_regression(x, y, data):
    X = data[[x]].values
    y_values = data[y].values
    X_train, X_test, y_train, y_test = train_test_split(X, y_values, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f'R^2: {r2:.3f}')
    sns.regplot(x=x, y=y, data=data, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.show()

# === Questão 1: Grid vs Posição Final ===
linear_regression('grid', 'positionOrder', results)

# === Questão 2: Grid vs Pontos ===
linear_regression('grid', 'points', results)

# === Questão 3: Posição Final vs Pontos ===
linear_regression('positionOrder', 'points', results)

# === Questão 4: Velocidade da Volta Mais Rápida vs Pontos ===
linear_regression('fastestLapSpeed', 'points', results)

# === Questão 5: Volta Mais Rápida vs Velocidade da Volta Mais Rápida ===
linear_regression('fastestLap', 'fastestLapSpeed', results)

# === Questão 6: Posição no Grid vs Volta Mais Rápida ===
linear_regression('grid', 'fastestLap', results)

# === Questão 7: Posição no Grid vs Voltas Completadas ===
linear_regression('grid', 'laps', results)

# === Questão 8: Voltas Completadas vs Tempo (ms) ===
linear_regression('laps', 'milliseconds', results)

# === Questão 9: Ranking da Volta Mais Rápida vs Velocidade ===
linear_regression('rank', 'fastestLapSpeed', results)

# === Questão 10: Grid e Velocidade da Volta Mais Rápida Prevendo Pontos ===
X = results[['grid', 'fastestLapSpeed']]
Y = results['points']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f'R^2: {r2_score(y_test, pred):.3f}')
sns.regplot(x=model.predict(X), y=Y, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.xlabel('Pontuação Prevista')
plt.ylabel('Pontuação Real')
plt.show()
