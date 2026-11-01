from fastapi import FastAPI
import matplotlib
# Importante para servidores sin entorno gráfico desplegado:
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Servicio de análisis activo"}

@app.get("/ejecutar-analisis")
def ejecutar_analisis():
    # Aquí ejecutas tu lógica de análisis o gráficos
    # Guardar gráfica en disco temporal si es necesario:
    # plt.savefig('grafica.png')
    # plt.close()
    return {"status": "exito", "mensaje": "Análisis completado"}