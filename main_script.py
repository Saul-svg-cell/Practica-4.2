import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LassoCV

# Configurar Matplotlib para entornos headless o backend por defecto
import matplotlib
matplotlib.use('Agg')  # Cambiar a plt.show() si estás trabajando localmente

# ==============================================================================
# PASO 1: SIMULACIÓN DE DATOS REALISTAS DE RIEGO AGRÍCOLA (SIM2)
# ==============================================================================

np.random.seed(42)
dias = 100

df_agri = pd.DataFrame({
    'radiacion_sol': np.random.uniform(12, 28, dias), # MJ/m2 al día
    'humedad_rel': np.random.uniform(40, 85, dias),   # % humedad
    'temperatura': np.random.uniform(15, 35, dias),   # °C temperatura
    'viento_vel': np.random.uniform(1, 5, dias)       # m/s velocidad viento
})

# Fórmula semi-empírica para la evapotranspiración (ETo)
df_agri['eto'] = (
    0.04 * df_agri['radiacion_sol'] +
    0.03 * (df_agri['temperatura'] - 15) -
    0.01 * (df_agri['humedad_rel'] - 50) +
    0.005 * df_agri['viento_vel'] +
    np.random.normal(0, 0.2, dias)
)

riego = np.random.uniform(2, 8, dias)
df_agri['riego'] = riego
df_agri['rendimiento'] = 0.8 * df_agri['eto'] + np.random.normal(0, 0.1, dias)

print("\n--- PASO 1: PRIMERAS 5 FILAS DEL DATASET CREADO ---")
print(df_agri.head())
print("-" * 50)

# ==============================================================================
# PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r)
# ==============================================================================

matriz_correlacion = df_agri.corr(method='pearson')

print("\n--- PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r) ---")
print(matriz_correlacion)

correlaciones_target = matriz_correlacion['eto'].sort_values(ascending=False)
print("\nCorrelaciones con 'eto':")
print(correlaciones_target)

# ==============================================================================
# PASO 3: REGRESIÓN LINEAL MÚLTIPLE CON SKLEARN
# ==============================================================================

X = df_agri[['radiacion_sol', 'temperatura', 'humedad_rel', 'viento_vel']]
y = df_agri['eto']

modelo_ols = LinearRegression()
modelo_ols.fit(X, y)

print("\n--- PASO 3: RESUMEN DEL MODELO OLS ---")
print(f"Intercepto: {modelo_ols.intercept_:.4f}")
for col, coef in zip(X.columns, modelo_ols.coef_):
    print(f"  {col}: {coef:.4f}")

# ==============================================================================
# PASO 4: REGRESIÓN LASSO (L1)
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

y_pred_lasso = lasso_cv.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print("\n--- PASO 4: REGRESIÓN LASSO ---")
print(f"Alpha elegido por CV: {lasso_cv.alpha_:.4f}")
print(f"R2 en test: {r2_lasso:.4f}")
print(f"RMSE en test: {rmse_lasso:.4f}")

# ==============================================================================
# PASO 5: REGRESIÓN NO LINEAL Y AJUSTE DE CURVA
# ==============================================================================

def funcion_no_lineal(x, a, b, c):
    return a * np.exp(b * x) + c

popt, _ = curve_fit(funcion_no_lineal, df_agri['temperatura'], df_agri['eto'])
y_pred_nl = funcion_no_lineal(df_agri['temperatura'], *popt)
r2_nl = r2_score(df_agri['eto'], y_pred_nl)

# ==============================================================================
# PASO 6: PANEL DE DIAGNÓSTICO VISUAL (GRÁFICAS INTEGRADAS)
# ==============================================================================

# Variables requeridas para las gráficas
y_pred_completo = modelo_ols.predict(X)
residuos = y - y_pred_completo

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Agrícola', fontsize=16, fontweight='bold')

# 1. Mapa de Calor de Correlación
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

# 2. Scatter Plot: Humedad vs ETo
sns.regplot(data=df_agri, x='humedad_rel', y='eto', color='b', ax=axes[0, 1],
            line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Relativa vs ETo')
axes[0, 1].set_xlabel('Humedad Relativa (%)')
axes[0, 1].set_ylabel('ETo (mm/día)')

# 3. Valores Reales vs Predichos
axes[0, 2].scatter(y_pred_completo, y, alpha=0.7, color='teal')
axes[0, 2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2_score(y, y_pred_completo)*100:.1f}%)')
axes[0, 2].set_xlabel('ETo Predicho (ŷ)')
axes[0, 2].set_ylabel('ETo Real (y)')

# 4. Residuos vs Valores Ajustados (Homocedasticidad)
axes[1, 0].scatter(y_pred_completo, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

# 5. Gráfico Q-Q Normal (Normalidad de Errores)
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

# Ocultar la celda sobrante [1, 2]
axes[1, 2].axis('off')

plt.tight_layout()

# Si ejecutas en local de forma interactiva:
# plt.show()

# Si ejecutas en un servidor/contenedor (como Render):
plt.savefig('diagnostico_regresion.png')
plt.close()
print("\nGráfica guardada como 'diagnostico_regresion.png'")
# 5. Gráfico Q-Q Normal (Normalidad de Errores)
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

# Ocultar la celda sobrante [1, 2]
axes[1, 2].axis('off')

plt.tight_layout()

# Si ejecutas en local de forma interactiva:
#plt.show()

# Si ejecutas en un servidor/contenedor (como Render):
plt.savefig('diagnostico_regresion.png')
plt.close()
print("\nGráfica guardada como 'diagnostico_regresion.png'")