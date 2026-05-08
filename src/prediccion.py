"""
Funciones de inferencia: cargar artefactos y calcular probabilidad + banda de riesgo.

La app Streamlit y cualquier otro servicio deberían importar desde aquí para no
duplicar lógica (principio DRY: Don't Repeat Yourself).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Tipos explícitos para lectura del código en clase.
BandaRiesgo = str


def _raiz_proyecto() -> Path:
    """Raíz del repo (sube dos niveles desde src/prediccion.py)."""
    return Path(__file__).resolve().parent.parent


def cargar_pipeline_modelo(carpeta_artifacts: Path | None = None) -> Pipeline:
    """
    Carga el archivo joblib del pipeline entrenado.

    Args:
        carpeta_artifacts: si es None, usa `artifacts/` en la raíz del proyecto.

    Returns:
        Pipeline de sklearn listo para `.predict_proba`.
    """
    raiz = _raiz_proyecto()
    base = carpeta_artifacts or (raiz / "artifacts")
    ruta = base / "modelo_pipeline.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta: python -m src.entrenar"
        )
    return joblib.load(ruta)


def cargar_umbrales(carpeta_artifacts: Path | None = None) -> dict[str, Any]:
    """Lee umbrales_riesgo.json con los cortes bajo/medio/alto."""
    raiz = _raiz_proyecto()
    base = carpeta_artifacts or (raiz / "artifacts")
    ruta = base / "umbrales_riesgo.json"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Ejecuta: python -m src.entrenar")
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_nombres_features(carpeta_artifacts: Path | None = None) -> list[str]:
    """Lista ordenada de nombres de columnas que espera el modelo."""
    raiz = _raiz_proyecto()
    base = carpeta_artifacts or (raiz / "artifacts")
    ruta = base / "features.json"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Ejecuta: python -m src.entrenar")
    data = json.loads(ruta.read_text(encoding="utf-8"))
    return list(data["features"])


def probabilidad_a_banda(probabilidad: float, umbrales: dict[str, Any]) -> tuple[BandaRiesgo, str]:
    """
    Convierte un número entre 0 y 1 en una etiqueta de negocio.

    Args:
        probabilidad: P(incumplimiento) devuelta por el modelo.
        umbrales: diccionario leído de umbrales_riesgo.json.

    Returns:
        (clave_banda, texto_legible) por ejemplo ("medio", "Riesgo medio").
    """
    bajo_hasta = float(umbrales["bajo_hasta"])
    medio_hasta = float(umbrales["medio_hasta"])
    etiquetas: dict[str, str] = umbrales["etiquetas"]

    if probabilidad < bajo_hasta:
        return "bajo", etiquetas["bajo"]
    if probabilidad < medio_hasta:
        return "medio", etiquetas["medio"]
    return "alto", etiquetas["alto"]


def predecir_fila(
    fila: dict[str, float | int],
    modelo: Pipeline | None = None,
    umbrales: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calcula la probabilidad de incumplimiento y la banda para un solo registro.

    Args:
        fila: diccionario con las mismas llaves que las columnas de entrenamiento.
        modelo: pipeline opcional (si ya lo cargaste en memoria para ir más rápido).
        umbrales: dict opcional de umbrales (misma lógica que modelo).

    Returns:
        Diccionario con probabilidad, banda, texto y los valores usados (trazabilidad).
    """
    if modelo is None:
        modelo = cargar_pipeline_modelo()
    if umbrales is None:
        umbrales = cargar_umbrales()

    nombres = cargar_nombres_features()
    faltan = [c for c in nombres if c not in fila]
    if faltan:
        raise ValueError(f"Faltan campos en la entrada: {faltan}")

    # DataFrame de una fila mantiene nombres de columnas; el pipeline es feliz.
    X = pd.DataFrame([{c: fila[c] for c in nombres}])

    proba = float(modelo.predict_proba(X)[0, 1])
    # Por seguridad numérica, recortamos al intervalo [0, 1].
    proba = float(np.clip(proba, 0.0, 1.0))

    clave_banda, texto_banda = probabilidad_a_banda(proba, umbrales)

    return {
        "probabilidad_incumplimiento": proba,
        "banda_clave": clave_banda,
        "banda_texto": texto_banda,
        "entradas_utilizadas": {c: fila[c] for c in nombres},
    }
