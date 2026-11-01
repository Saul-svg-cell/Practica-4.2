# Imagen base liviana
FROM python:3.11-slim

# Evita la creación de archivos .pyc y fuerza salida sin búfer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar/graficar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Puerto por el que escuchará FastAPI
EXPOSE 10000

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]