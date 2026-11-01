import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LassoCV

# Inicializar la aplicación FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Servicio de análisis activo"}

@app.get("/ejecutar-analisis")
def ejecutar_analisis():
    # 1. Simulación de datos
    np.random.seed(42)
    dias = 100
    df_agri = pd.DataFrame({
        'radiacion_sol': np.random.uniform(12, 28, dias),
        'humedad_rel': np.random.uniform(40, 85, dias),
        'temperatura': np.random.uniform(15, 35, dias),
        'viento_vel': np.random.uniform(1, 5, dias)
    })
    
    df_agri['eto'] = (
        0.04 * df_agri['radiacion_sol'] +
        0.03 * (df_agri['temperatura'] - 15) -
        0.01 * (df_agri['humedad_rel'] - 50) +
        0.005 * df_agri['viento_vel'] +
        np.random.normal(0, 0.2, dias)
    )

    # 2. Modelo OLS
    X = df_agri[['radiacion_sol', 'temperatura', 'humedad_rel', 'viento_vel']]
    y = df_agri['eto']
    modelo_ols = LinearRegression()
    modelo_ols.fit(X, y)

    # 3. Lasso CV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    y_pred_lasso = lasso_cv.predict(X_test)
    r2_lasso = float(r2_score(y_test, y_pred_lasso))
    rmse_lasso = float(np.sqrt(mean_squared_error(y_test, y_pred_lasso)))

    # 4. Retornar los resultados en formato JSON/Texto
    return {
        "status": "exito",
        "modelo_ols": {
            "intercepto": round(float(modelo_ols.intercept_), 4),
            "coeficientes": {col: round(float(coef), 4) for col, coef in zip(X.columns, modelo_ols.coef_)}
        },
        "modelo_lasso": {
            "alpha_optimo": round(float(lasso_cv.alpha_), 4),
            "r2_test": round(r2_lasso, 4),
            "rmse_test": round(rmse_lasso, 4),
            "coeficientes": {col: round(float(coef), 4) for col, coef in zip(X.columns, lasso_cv.coef_)}
        }
    }