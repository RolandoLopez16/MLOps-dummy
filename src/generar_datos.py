"""
Generación de un dataset 100 % sintético de créditos para el laboratorio.

No usamos datos reales de personas: todo es simulado con reglas simples
para que el alumno pueda practicar el flujo MLOps sin preocuparse por privacidad.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Semilla fija: al ejecutar dos veces el script, salen los mismos números (reproducibilidad).
SEMILLA = 42


def _ruta_proyecto() -> Path:
    """Devuelve la carpeta raíz del proyecto (donde está la carpeta `data/`)."""
    return Path(__file__).resolve().parent.parent


def generar_dataframe_creditos(n_filas: int = 2500) -> pd.DataFrame:
    """
    Crea un DataFrame con variables que se parecen a un score de riesgo crediticio.

    La etiqueta `incumplimiento` (0 = paga bien, 1 = incumple) se construye con
    una fórmula tipo “score” + ruido, para que un modelo de ML pueda aprender
    patrones sin que el problema sea trivial.
    """
    rng = np.random.default_rng(SEMILLA)

    # Ingreso mensual en pesos colombianos aproximados (orden de magnitud realista para ejercicio).
    ingreso_mensual = rng.lognormal(mean=np.log(2_500_000), sigma=0.45, size=n_filas)

    # Deuda total también en escala similar al ingreso.
    deuda_total = rng.lognormal(mean=np.log(1_800_000), sigma=0.55, size=n_filas)

    # Meses en el empleo actual.
    antiguedad_laboral_meses = rng.integers(low=0, high=240, size=n_filas)

    # Cantidad de productos de crédito activos.
    num_creditos_activos = rng.integers(low=0, high=8, size=n_filas)

    # Número de moras reportadas en los últimos 12 meses.
    moras_ult_12m = rng.poisson(lam=0.8, size=n_filas)
    moras_ult_12m = np.clip(moras_ult_12m, 0, 12)

    # --- Construcción de la etiqueta (incumplimiento) de forma sintética ---
    # Relación deuda/ingreso: a mayor carga, más riesgo.
    ratio_deuda_ingreso = deuda_total / np.maximum(ingreso_mensual, 1.0)

    # “Score” lineal en el espacio log-odds; coeficientes solo didácticos.
    # El término constante (-2.25) se calibró para que la tasa de incumplimiento
    # quede alrededor del 40 % (más hablable en clase que tasas muy extremas).
    z = (
        -2.25
        + 2.1 * ratio_deuda_ingreso
        + 0.04 * moras_ult_12m
        + 0.12 * num_creditos_activos
        - 0.003 * antiguedad_laboral_meses
        + rng.normal(0.0, 0.35, size=n_filas)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    incumplimiento = (rng.uniform(size=n_filas) < prob).astype(int)

    df = pd.DataFrame(
        {
            "ingreso_mensual": np.round(ingreso_mensual, 2),
            "deuda_total": np.round(deuda_total, 2),
            "antiguedad_laboral_meses": antiguedad_laboral_meses,
            "num_creditos_activos": num_creditos_activos,
            "moras_ult_12m": moras_ult_12m,
            "incumplimiento": incumplimiento,
        }
    )
    return df


def guardar_csv_y_metadatos(df: pd.DataFrame, carpeta_data: Path) -> None:
    """
    Escribe el CSV principal y un pequeño JSON con la descripción de columnas.

    El JSON sirve para documentar qué significa cada variable en el README o en clase.
    """
    carpeta_data.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta_data / "creditos_sinteticos.csv"
    df.to_csv(ruta_csv, index=False)

    meta = {
        "descripcion": "Dataset sintético BancoDeTodos — solo para aprendizaje.",
        "columnas": {
            "ingreso_mensual": "Ingreso mensual aproximado (moneda ficticia del ejercicio).",
            "deuda_total": "Suma aproximada de deudas vigentes.",
            "antiguedad_laboral_meses": "Meses en el empleo actual.",
            "num_creditos_activos": "Cantidad de líneas de crédito activas.",
            "moras_ult_12m": "Número de moras en los últimos 12 meses.",
            "incumplimiento": "1 si el cliente incumple en el escenario simulado, 0 si no.",
        },
        "semilla": SEMILLA,
        "n_filas": len(df),
    }
    (carpeta_data / "diccionario_datos.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """Punto de entrada cuando ejecutas: python -m src.generar_datos"""
    raiz = _ruta_proyecto()
    carpeta_data = raiz / "data"
    print("Generando datos sintéticos...")
    df = generar_dataframe_creditos()
    guardar_csv_y_metadatos(df, carpeta_data)
    print(f"Listo. Archivo guardado en: {carpeta_data / 'creditos_sinteticos.csv'}")
    print(f"Filas: {len(df)}, proporción incumplimiento: {df['incumplimiento'].mean():.2%}")


if __name__ == "__main__":
    main()
