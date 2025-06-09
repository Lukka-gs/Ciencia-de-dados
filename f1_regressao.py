import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Carrega o dataset com tratamento para valores faltantes '\\N'
df = pd.read_csv('archive/results.csv', na_values='\\N')

# Funcao auxiliar para executar regressao linear simples e exibir o grafico
def regressao_linear(x_col: str, y_col: str, data: pd.DataFrame):
    """Realiza regressao linear simples entre x_col e y_col."""
    subset = data[[x_col, y_col]].dropna()
    X = subset[[x_col]].values
    y = subset[y_col].values

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)

    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=x_col, y=y_col, data=subset, alpha=0.5)
    plt.plot(subset[x_col], y_pred, color='red')
    plt.title(f'{y_col} vs {x_col}')
    plt.tight_layout()
    plt.show()
    plt.close()
    print(f'Coeficiente de Determinacao (R^2): {r2:.4f}\n')

# === Questao 1 ===
# Relacao entre grid e positionOrder
regressao_linear('grid', 'positionOrder', df)

# === Questao 2 ===
# Relacao entre grid e fastestLap
regressao_linear('grid', 'fastestLap', df)

# === Questao 3 ===
# Regressao linear entre grid (largada) e points
regressao_linear('grid', 'points', df)

# === Questao 4 ===
# Relacao entre laps e points
regressao_linear('laps', 'points', df)

# === Questao 5 ===
# Relacao entre fastestLap e points
regressao_linear('fastestLap', 'points', df)

# === Questao 6 ===
# Relacao entre fastestLapSpeed e points
regressao_linear('fastestLapSpeed', 'points', df)

# === Questao 7 ===
# Relacao entre constructorId e points
regressao_linear('constructorId', 'points', df)

# === Questao 8 ===
# Relacao entre driverId e points
regressao_linear('driverId', 'points', df)

# === Questao 9 ===
# Relacao entre raceId e points
regressao_linear('raceId', 'points', df)

# === Questao 10 ===
# Relacao entre number e points
regressao_linear('number', 'points', df)
