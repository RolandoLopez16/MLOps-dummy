"""
Script de entrenamiento reproducible.

Lee el CSV sintético, entrena un pipeline sencillo (escalado + regresión logística)
y guarda artefactos en `artifacts/` para que la app Streamlit pueda cargarlos.

Los estudiantes deben entender: entrenar una vez ≠ desplegar; el despliegue solo
consume los archivos guardados aquí.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Mismas columnas que usa la app (orden fijo: evita errores al predecir).
COLUMNAS_FEATURES = [
    "ingreso_mensual",
    "deuda_total",
    "antiguedad_laboral_meses",
    "num_creditos_activos",
    "moras_ult_12m",
]
COLUMNA_OBJETIVO = "incumplimiento"


def _raiz_proyecto() -> Path:
    """Ubica la carpeta raíz del repositorio."""
    return Path(__file__).resolve().parent.parent


def cargar_datos(ruta_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga el CSV y separa características (X) de la etiqueta (y).

    Returns:
        X: solo las columnas de entrada del modelo.
        y: columna binaria incumplimiento.
    """
    df = pd.read_csv(ruta_csv)
    faltan = [c for c in COLUMNAS_FEATURES + [COLUMNA_OBJETIVO] if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}")
    X = df[COLUMNAS_FEATURES]
    y = df[COLUMNA_OBJETIVO]
    return X, y


def construir_pipeline() -> Pipeline:
    """
    Define el modelo como una Pipeline de scikit-learn.

    - StandardScaler: lleva cada variable numérica a media 0 y varianza 1.
      Ayuda a que la regresión logística converja bien.
    - LogisticRegression: modelo lineal simple, fácil de explicar en clase.

    max_iter sube un poco por si el dataset simulado es “duro” para el optimizador.
    """
    return Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            (
                "clasificador",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )


def guardar_umbrales(carpeta_artifacts: Path) -> None:
    """
    Guarda los cortes de probabilidad para las bandas de riesgo en la app.

    Estos valores son una decisión de negocio (no vienen del ML): las probabilidades
    salen del modelo; las etiquetas “bajo/medio/alto” las definimos nosotros.
    """
    umbrales = {
        "bajo_hasta": 0.30,
        "medio_hasta": 0.60,
        "etiquetas": {
            "bajo": "Riesgo bajo",
            "medio": "Riesgo medio",
            "alto": "Riesgo alto",
        },
    }
    ruta = carpeta_artifacts / "umbrales_riesgo.json"
    ruta.write_text(json.dumps(umbrales, ensure_ascii=False, indent=2), encoding="utf-8")


def guardar_lista_features(carpeta_artifacts: Path) -> None:
    """Guarda el orden exacto de columnas que espera el modelo al predecir."""
    ruta = carpeta_artifacts / "features.json"
    ruta.write_text(
        json.dumps({"features": COLUMNAS_FEATURES}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Ejecutar: python -m src.entrenar  (desde la raíz del proyecto)."""
    raiz = _raiz_proyecto()
    ruta_csv = raiz / "data" / "creditos_sinteticos.csv"
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No existe {ruta_csv}. Ejecuta antes: python -m src.generar_datos"
        )

    carpeta_artifacts = raiz / "artifacts"
    carpeta_artifacts.mkdir(parents=True, exist_ok=True)

    X, y = cargar_datos(ruta_csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modelo = construir_pipeline()
    modelo.fit(X_train, y_train)

    # Probabilidad de la clase positiva (incumplimiento = 1).
    prob_test = modelo.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, prob_test)
    acc = accuracy_score(y_test, pred_test)
    print(f"Métricas en prueba (solo referencia didáctica): AUC={auc:.3f}, accuracy={acc:.3f}")

    # Un solo archivo con todo el pipeline entrenado (escalado + modelo).
    ruta_modelo = carpeta_artifacts / "modelo_pipeline.joblib"
    joblib.dump(modelo, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

    guardar_umbrales(carpeta_artifacts)
    guardar_lista_features(carpeta_artifacts)
    print(f"Umbrales y lista de features en: {carpeta_artifacts}")


if __name__ == "__main__":
    main()
