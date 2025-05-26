from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load("modelo_gaussianNB_adventure.pkl")

# Crear app FastAPI
app = FastAPI()

# Modelo de entrada
class EntradaModelo(BaseModel):
    year_of_release: float

# Modelo de salida
class SalidaModelo(BaseModel):
    ventas_altas: int
    probabilidad: float

# Endpoint raíz (opcional)
@app.get("/")
def leer_root():
    return {"mensaje": "API de predicción de ventas altas para juegos Adventure"}

# Endpoint de predicción
@app.post("/proceso", response_model=SalidaModelo)
def predecir_ventas_altas(datos: EntradaModelo):
    X = np.array([[datos.year_of_release]])
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]
    return {
        "ventas_altas": int(pred),
        "probabilidad": round(float(prob), 3)
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://miapp.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)