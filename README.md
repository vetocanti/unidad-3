# Modelo de prediccion del clima con AutoML

Este proyecto entrena y usa un modelo de Machine Learning para predecir una categoria general del clima:

- `soleado`
- `nublado`
- `lluvioso`
- `frio`
- `caluroso`

El flujo general es:

1. Generar un dataset climatico de ejemplo.
2. Entrenar varios modelos automaticamente.
3. Optimizar hiperparametros con validacion cruzada.
4. Guardar el mejor modelo en `modelo_clima.pkl`.
5. Consultar datos climaticos desde OpenWeatherMap o usar datos de prueba.
6. Cargar el modelo guardado y predecir el clima.

## Archivos del proyecto

### `entrenar_modelo_clima.py`

Este archivo entrena el modelo. Es el script principal para aplicar AutoML y optimizacion de hiperparametros.

### `ejemplo_clima.py`

Este archivo usa el modelo ya entrenado para hacer una prediccion. Puede usar datos reales desde OpenWeatherMap si se configura una API key.

### `modelo_clima.pkl`

Este archivo se genera automaticamente al ejecutar el entrenamiento. Contiene el mejor modelo encontrado, las columnas usadas y los mejores hiperparametros.

## Requisitos

El proyecto usa estas librerias:

```bash
pip install -r requirements.txt
```


## Paso 1: entrenar el modelo

Ejecuta:

```bash
python entrenar_modelo_clima.py
```

Este comando:

1. Crea datos climaticos sinteticos.
2. Divide los datos en entrenamiento y prueba.
3. Prueba varios algoritmos.
4. Busca la mejor combinacion de hiperparametros.
5. Guarda el mejor modelo en `modelo_clima.pkl`.
6. Muestra metricas de evaluacion.

## Paso 2: usar el modelo para predecir

Ejecuta:

```bash
python ejemplo_clima.py
```

Si no configuraste una API key, el programa usa datos de ejemplo para comprobar que el modelo funciona.

## Paso 3: usar datos reales de OpenWeatherMap

Primero consigue una API key en:

```text
https://home.openweathermap.org/api_keys
```

Luego crea un archivo llamado `.env` en la carpeta del proyecto con este contenido:

```env
OPENWEATHER_API_KEY=tu_api_key
```

Despues ejecuta:

```bash
python ejemplo_clima.py
```

El archivo `ejemplo_clima.py` carga el archivo `.env` y lee la clave con:

```python
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY", "TU_API_KEY")
```

Esto evita dejar la clave escrita directamente dentro del codigo.

## Paso 4: usar la API con FastAPI

Ejecuta:

```bash
uvicorn api_clima:app --reload
```

Luego abre la documentacion interactiva:

```text
http://127.0.0.1:8000/docs
```

Endpoints principales:

- `GET /salud`: comprueba que el modelo se puede cargar.
- `GET /predecir?ciudad=Madrid`: consulta OpenWeatherMap y devuelve la prediccion del modelo.

## Explicacion de `entrenar_modelo_clima.py`

### 1. Importaciones

```python
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
```

Estas librerias se usan para:

- `joblib`: guardar y cargar el modelo entrenado.
- `numpy`: generar datos numericos aleatorios.
- `pandas`: trabajar con tablas de datos.
- `randint` y `uniform`: definir rangos de hiperparametros aleatorios.

Tambien se importan modelos y herramientas de `scikit-learn`:

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `LogisticRegression`
- `SVC`
- `KNeighborsClassifier`
- `RandomizedSearchCV`
- `train_test_split`
- `Pipeline`
- `StandardScaler`
- `accuracy_score`
- `classification_report`

### 2. Constantes principales

```python
RANDOM_STATE = 42
MODEL_PATH = "modelo_clima.pkl"
FEATURES = ["temperatura", "humedad", "presion", "viento", "nubosidad", "lluvia_1h"]
N_ITER_SEARCH = 30
CV_FOLDS = 5
```

Significado:

- `RANDOM_STATE`: permite repetir resultados similares cada vez que se ejecuta.
- `MODEL_PATH`: nombre del archivo donde se guarda el modelo.
- `FEATURES`: columnas que usara el modelo para aprender.
- `N_ITER_SEARCH`: cantidad de combinaciones de hiperparametros que se probaran.
- `CV_FOLDS`: cantidad de divisiones para validacion cruzada.

### 3. Funcion `clasificar_clima`

```python
def clasificar_clima(row):
```

Esta funcion crea la etiqueta o resultado esperado del modelo. Recibe una fila con datos climaticos y devuelve una categoria.

Reglas usadas:

- Si hay mucha lluvia o humedad alta con muchas nubes, clasifica como `lluvioso`.
- Si hay muchas nubes o humedad alta, clasifica como `nublado`.
- Si la temperatura es baja y hay humedad alta, clasifica como `frio`.
- Si la temperatura es alta, poca humedad y pocas nubes, clasifica como `caluroso`.
- Si no cumple lo anterior, clasifica como `soleado`.

Esta parte simula las etiquetas porque no estamos usando un dataset historico real.

### 4. Funcion `crear_dataset`

```python
def crear_dataset(cantidad=3000):
```

Genera un dataset sintetico de 3000 registros.

Crea estas variables:

- `temperatura`
- `humedad`
- `presion`
- `viento`
- `nubosidad`
- `lluvia_1h`

Luego aplica `clasificar_clima` para crear la columna objetivo:

```python
datos["clima"] = datos.apply(clasificar_clima, axis=1)
```

La columna `clima` es lo que el modelo intentara aprender a predecir.

### 5. Funcion `entrenar`

```python
def entrenar():
```

Esta funcion ejecuta todo el proceso de entrenamiento.

Primero crea el dataset:

```python
dataset = crear_dataset()
```

Despues separa:

```python
x = dataset[FEATURES]
y = dataset["clima"]
```

- `x`: variables de entrada.
- `y`: resultado esperado.

### 6. Division en entrenamiento y prueba

```python
x_train, x_test, y_train, y_test = train_test_split(...)
```

El dataset se divide en:

- 80% para entrenar.
- 20% para probar.

El parametro `stratify=y` mantiene una proporcion similar de clases en entrenamiento y prueba.

### 7. Pipeline

```python
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)
```

Un `Pipeline` permite unir pasos en orden:

1. `StandardScaler`: normaliza las columnas numericas.
2. `classifier`: modelo de clasificacion.

Aunque el clasificador inicial es `RandomForestClassifier`, durante el AutoML se reemplaza por otros modelos.

### 8. AutoML basico

El bloque `espacios_busqueda` define los modelos que se van a probar:

- Random Forest
- Gradient Boosting
- Regresion Logistica
- Support Vector Machine
- K Nearest Neighbors

Cada modelo tiene hiperparametros distintos. Por ejemplo, para Random Forest:

```python
"classifier__n_estimators": randint(80, 260)
"classifier__max_depth": randint(4, 18)
```

Esto significa que el entrenamiento probara automaticamente distintas cantidades de arboles y profundidades.

### 9. Optimizacion de hiperparametros

```python
automl = RandomizedSearchCV(...)
```

`RandomizedSearchCV` busca combinaciones aleatorias de modelos e hiperparametros.

Parametros importantes:

- `n_iter=N_ITER_SEARCH`: prueba 30 combinaciones.
- `cv=CV_FOLDS`: usa validacion cruzada de 5 partes.
- `scoring="f1_weighted"`: selecciona el mejor modelo segun F1 ponderado.
- `n_jobs=-1`: usa todos los nucleos disponibles del procesador.
- `verbose=1`: muestra informacion del entrenamiento.

### 10. Entrenamiento del AutoML

```python
automl.fit(x_train, y_train)
```

Aqui se entrenan todas las combinaciones seleccionadas.

Despues se obtiene el mejor modelo:

```python
modelo = automl.best_estimator_
```

### 11. Evaluacion

```python
predicciones = modelo.predict(x_test)
```

El modelo predice sobre datos que no vio durante el entrenamiento.

Luego se imprimen metricas:

```python
accuracy_score(y_test, predicciones)
classification_report(y_test, predicciones)
```

El reporte muestra:

- `precision`
- `recall`
- `f1-score`
- `support`

### 12. Guardado del modelo

```python
joblib.dump(
    {
        "modelo": modelo,
        "features": FEATURES,
        "mejor_score_cv": automl.best_score_,
        "mejores_hiperparametros": automl.best_params_,
    },
    MODEL_PATH,
)
```

Se guarda un diccionario con:

- El mejor modelo entrenado.
- Las columnas que espera el modelo.
- El mejor puntaje de validacion cruzada.
- Los mejores hiperparametros encontrados.

## Explicacion de `ejemplo_clima.py`

### 1. Importaciones

```python
from pathlib import Path
import os
import joblib
import pandas as pd
import requests
```

Se usan para:

- `Path`: manejar rutas de archivos.
- `os`: leer variables de entorno.
- `joblib`: cargar el modelo.
- `pandas`: construir el DataFrame para predecir.
- `requests`: consultar la API de clima.

### 2. Configuracion

```python
API_KEY = os.getenv("OPENWEATHER_API_KEY", "TU_API_KEY")
CITY = "Madrid"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
MODEL_PATH = Path("modelo_clima.pkl")
```

Aqui se configura:

- La API key.
- La ciudad.
- La URL de OpenWeatherMap.
- La ruta del modelo entrenado.

### 3. Funcion `procesar_datos`

```python
def procesar_datos(api_data):
```

Recibe la respuesta JSON de OpenWeatherMap y extrae solo las columnas que el modelo necesita:

- Temperatura.
- Humedad.
- Presion.
- Viento.
- Nubosidad.
- Lluvia en la ultima hora.

Devuelve un `DataFrame` con una sola fila:

```python
return pd.DataFrame([weather_data])
```

### 4. Funcion `cargar_modelo`

```python
def cargar_modelo():
```

Verifica si existe `modelo_clima.pkl`.

Si no existe, muestra un error indicando que primero se debe entrenar:

```bash
python entrenar_modelo_clima.py
```

Si existe, carga el archivo:

```python
paquete = joblib.load(MODEL_PATH)
```

Y retorna:

- El modelo.
- Las columnas esperadas.

### 5. Funcion `predecir_clima`

```python
def predecir_clima(datos, modelo, features):
```

Recibe:

- `datos`: DataFrame con datos climaticos.
- `modelo`: modelo entrenado.
- `features`: columnas que el modelo necesita.

Hace la prediccion:

```python
prediccion = modelo.predict(datos[features])
```

Y devuelve una categoria como `nublado` o `soleado`.

### 6. Funcion `obtener_datos_clima`

```python
def obtener_datos_clima():
```

Si no existe API key, usa datos de ejemplo:

```python
if API_KEY == "TU_API_KEY":
```

Esto permite probar el programa sin conectarse a internet.

Si hay API key, consulta OpenWeatherMap:

```python
response = requests.get(URL, timeout=10)
response.raise_for_status()
return response.json()
```

### 7. Funcion `main`

```python
def main():
```

Es el flujo principal del programa:

1. Carga el modelo.
2. Obtiene datos climaticos.
3. Procesa los datos.
4. Muestra los datos procesados.
5. Predice el clima.
6. Imprime el resultado.

### 8. Punto de entrada

```python
if __name__ == "__main__":
    main()
```

Esto hace que `main()` se ejecute solo cuando el archivo se corre directamente con:

```bash
python ejemplo_clima.py
```

## Que significa AutoML en este proyecto

AutoML significa automatizar partes del proceso de Machine Learning.

En este proyecto se automatiza:

1. La comparacion entre varios algoritmos.
2. La busqueda de hiperparametros.
3. La seleccion del mejor modelo.

No es una plataforma AutoML completa como Auto-sklearn, FLAML o TPOT, pero cumple el objetivo educativo usando `scikit-learn`.

## Que son los hiperparametros

Los hiperparametros son configuraciones del modelo que se deciden antes de entrenar.

Ejemplos:

- Cantidad de arboles en Random Forest.
- Profundidad maxima de los arboles.
- Tasa de aprendizaje en Gradient Boosting.
- Numero de vecinos en KNN.
- Kernel y valor `C` en SVM.

El codigo usa `RandomizedSearchCV` para probar combinaciones y quedarse con la mejor.

## Limitaciones del proyecto

Este proyecto usa datos sinteticos. Eso significa que sirve para aprender el flujo de trabajo, pero no representa un modelo meteorologico profesional.

Para mejorar el proyecto se podria:

1. Usar datos historicos reales.
2. Agregar fecha, hora, estacion del ano y ubicacion geografica.
3. Predecir valores numericos, como temperatura futura o probabilidad de lluvia.
4. Guardar resultados de entrenamiento en un archivo CSV.
5. Crear una interfaz web o una API con Flask/FastAPI.
