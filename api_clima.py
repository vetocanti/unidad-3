from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from requests import HTTPError, RequestException

from ejemplo_clima import cargar_modelo, obtener_datos_clima, predecir_clima, procesar_datos


app = FastAPI(
    title="API Predictor de Clima",
    description="API para predecir una categoria de clima usando OpenWeatherMap y un modelo entrenado.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def obtener_modelo_cacheado():
    return cargar_modelo()


@app.get("/")
def inicio():
    return {
        "mensaje": "API Predictor de Clima",
        "endpoints": {
            "salud": "/salud",
            "prediccion": "/predecir?ciudad=Madrid",
            "documentacion": "/docs",
        },
    }


@app.get("/salud")
def salud():
    try:
        obtener_modelo_cacheado()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"estado": "ok", "modelo": "cargado"}


@app.get("/predecir")
def predecir(
    ciudad: str = Query(
        default="Madrid",
        min_length=1,
        description="Ciudad para consultar en OpenWeatherMap.",
    )
):
    try:
        modelo, features = obtener_modelo_cacheado()
        api_data = obtener_datos_clima(ciudad)
        datos = procesar_datos(api_data)
        prediccion = predecir_clima(datos, modelo, features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(
            status_code=status_code,
            detail="Error al consultar OpenWeatherMap. Revisa la ciudad o la API key.",
        ) from exc
    except RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="No se pudo conectar con OpenWeatherMap.",
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"La respuesta climatica no tiene el campo esperado: {exc}",
        ) from exc

    clima_actual = api_data.get("weather", [{}])[0]
    fila = datos.iloc[0].to_dict()

    return {
        "ciudad": api_data.get("name", ciudad),
        "prediccion": prediccion,
        "datos_modelo": fila,
        "clima_actual": {
            "descripcion": clima_actual.get("description"),
            "codigo": clima_actual.get("id"),
        },
    }
