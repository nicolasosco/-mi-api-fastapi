from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelo entrenado
modelo = joblib.load("modelo_gaussianNB_adventure.pkl")

# Inicializar FastAPI
app = FastAPI()

# Esquema de entrada con Pydantic
class EntradaModelo(BaseModel):
    year_of_release: float

# Endpoint para hacer predicciones
@app.post("/proceso")
def predecir_ventas_altas(datos: EntradaModelo):
    entrada = np.array([[datos.year_of_release]])
    prediccion = modelo.predict(entrada)[0]
    probabilidad = modelo.predict_proba(entrada)[0][1]
    return {
        "ventas_altas": int(prediccion),
        "probabilidad": round(probabilidad, 3)
    }

