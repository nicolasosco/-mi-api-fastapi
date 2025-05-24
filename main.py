from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load("modelo_gaussianNB_adventure.pkl")

# Crear app FastAPI
app = FastAPI()

# Definir la estructura del input
class InputData(BaseModel):
    Year_of_Release: int

# Endpoint raíz
@app.get("/")
def read_root():
    return {"mensaje": "API de predicción de ventas altas para juegos Adventure"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: InputData):
    X_input = np.array([[data.Year_of_Release]])
    pred = modelo.predict(X_input)[0]
    prob = modelo.predict_proba(X_input)[0][1]
    return {
        "prediccion": int(pred),
        "probabilidad_venta_alta": round(float(prob), 4)
    }
    
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Modelo de entrada
class EntradaModelo(BaseModel):
    year_of_release: int

# Modelo de salida
class SalidaModelo(BaseModel):
    resultado: bool

@app.post("/proceso", response_model=SalidaModelo)
def predecir_ventas_altas(datos: EntradaModelo):
    # Lógica de predicción (simulada aquí como ejemplo)
    prediccion = datos.year_of_release > 2010
    return {"resultado": prediccion}

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

# Endpoint
@app.post("/proceso", response_model=SalidaModelo)
def predecir_ventas_altas(datos: EntradaModelo):
    X = np.array([[datos.year_of_release]])
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]
    return {
        "ventas_altas": int(pred),
        "probabilidad": float(round(prob, 3))
    }

