import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
MODEL_PATH = "modelo_clima.pkl"
FEATURES = ["temperatura", "humedad", "presion", "viento", "nubosidad", "lluvia_1h"]
N_ITER_SEARCH = 30
CV_FOLDS = 5


def clasificar_clima(row):
    """Etiqueta aproximada para crear datos de entrenamiento de ejemplo."""
    if row["lluvia_1h"] >= 3 or (row["humedad"] >= 82 and row["nubosidad"] >= 70):
        return "lluvioso"
    if row["nubosidad"] >= 65 or row["humedad"] >= 75:
        return "nublado"
    if row["temperatura"] <= 5 and row["humedad"] >= 70:
        return "frio"
    if row["temperatura"] >= 28 and row["humedad"] <= 65 and row["nubosidad"] <= 45:
        return "caluroso"
    return "soleado"


def crear_dataset(cantidad=3000):
    """Genera un dataset sintetico para entrenar el modelo sin depender de una API."""
    rng = np.random.default_rng(RANDOM_STATE)
    datos = pd.DataFrame(
        {
            "temperatura": rng.normal(18, 10, cantidad).clip(-8, 40),
            "humedad": rng.normal(62, 22, cantidad).clip(10, 100),
            "presion": rng.normal(1013, 12, cantidad).clip(970, 1045),
            "viento": rng.gamma(2.2, 2.4, cantidad).clip(0, 35),
            "nubosidad": rng.uniform(0, 100, cantidad),
        }
    )

    prob_lluvia = (
        (datos["humedad"] / 100) * 0.45
        + (datos["nubosidad"] / 100) * 0.45
        + np.where(datos["presion"] < 1005, 0.15, 0)
    ).clip(0, 0.95)
    datos["lluvia_1h"] = np.where(
        rng.random(cantidad) < prob_lluvia,
        rng.gamma(1.5, 2.0, cantidad),
        0,
    ).clip(0, 30)

    datos["clima"] = datos.apply(clasificar_clima, axis=1)
    return datos


def entrenar():
    dataset = crear_dataset()
    x = dataset[FEATURES]
    y = dataset["clima"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )

    espacios_busqueda = [
        {
            "classifier": [RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")],
            "classifier__n_estimators": randint(80, 260),
            "classifier__max_depth": randint(4, 18),
            "classifier__min_samples_split": randint(2, 12),
            "classifier__min_samples_leaf": randint(1, 8),
        },
        {
            "classifier": [GradientBoostingClassifier(random_state=RANDOM_STATE)],
            "classifier__n_estimators": randint(60, 220),
            "classifier__learning_rate": uniform(0.03, 0.22),
            "classifier__max_depth": randint(2, 7),
        },
        {
            "classifier": [LogisticRegression(max_iter=1500, class_weight="balanced")],
            "classifier__C": uniform(0.1, 8),
            "classifier__solver": ["lbfgs"],
        },
        {
            "classifier": [SVC(class_weight="balanced")],
            "classifier__C": uniform(0.2, 8),
            "classifier__gamma": ["scale", "auto"],
            "classifier__kernel": ["rbf", "linear"],
        },
        {
            "classifier": [KNeighborsClassifier()],
            "classifier__n_neighbors": randint(3, 18),
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
    ]

    automl = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=espacios_busqueda,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    automl.fit(x_train, y_train)
    modelo = automl.best_estimator_
    predicciones = modelo.predict(x_test)

    joblib.dump(
        {
            "modelo": modelo,
            "features": FEATURES,
            "mejor_score_cv": automl.best_score_,
            "mejores_hiperparametros": automl.best_params_,
        },
        MODEL_PATH,
    )

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Mejor score CV F1 ponderado: {automl.best_score_:.2%}")
    print("Mejor configuracion encontrada:")
    for parametro, valor in automl.best_params_.items():
        print(f"  {parametro}: {valor}")
    print(f"Exactitud de prueba: {accuracy_score(y_test, predicciones):.2%}")
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, predicciones))


if __name__ == "__main__":
    entrenar()
