from pathlib import Path
import os

from dotenv import load_dotenv
import joblib
import pandas as pd
import requests


load_dotenv()

# Configura tu clave API y ciudad.
API_KEY = os.getenv("OPENWEATHER_API_KEY", "TU_API_KEY")
CITY = "Madrid"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
MODEL_PATH = Path("modelo_clima.pkl")


def procesar_datos(api_data):
    """Procesa los datos de OpenWeatherMap y crea un DataFrame para el modelo."""
    main_data = api_data["main"]
    wind_data = api_data.get("wind", {})
    clouds_data = api_data.get("clouds", {})
    rain_data = api_data.get("rain", {})

    weather_data = {
        "temperatura": main_data["temp"],
        "humedad": main_data["humidity"],
        "presion": main_data["pressure"],
        "viento": wind_data.get("speed", 0),
        "nubosidad": clouds_data.get("all", 0),
        "lluvia_1h": rain_data.get("1h", 0),
    }
    return pd.DataFrame([weather_data])


def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No existe modelo_clima.pkl. Ejecuta primero: python entrenar_modelo_clima.py"
        )

    paquete = joblib.load(MODEL_PATH)
    return paquete["modelo"], paquete["features"]


def predecir_clima(datos, modelo, features):
    prediccion = modelo.predict(datos[features])
    return prediccion[0]


def obtener_datos_clima(ciudad=CITY):
    if API_KEY == "TU_API_KEY":
        print("No configuraste una API key. Usando datos de ejemplo para probar el modelo.")
        return {
            "name": ciudad,
            "main": {"temp": 19, "humidity": 78, "pressure": 1010},
            "wind": {"speed": 4.2},
            "clouds": {"all": 80},
            "rain": {"1h": 1.1},
        }

    params = {"q": ciudad, "appid": API_KEY, "units": "metric"}
    response = requests.get(OPENWEATHER_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def main():
    print("Obteniendo datos climaticos...")
    modelo, features = cargar_modelo()
    api_data = obtener_datos_clima(CITY)
    datos_procesados = procesar_datos(api_data)

    print("Datos procesados para el modelo:")
    print(datos_procesados)

    prediccion = predecir_clima(datos_procesados, modelo, features)
    print(f"Prediccion del clima para {CITY}: {prediccion}")


if __name__ == "__main__":
    main()
