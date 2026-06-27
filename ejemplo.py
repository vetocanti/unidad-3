import requests
import pandas as pd

# Configura tu clave API y ciudad
API_KEY = "TU_API_KEY"
CITY = "Madrid"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

# Consumir la API
response = requests.get(URL)
if response.status_code == 200:
    data = response.json()
    print("Datos recibidos:", data)
else:
    print("Error al conectar con la API:", response.status_code)
