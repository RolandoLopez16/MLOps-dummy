"""
Aplicación web de demostración — BancoDeTodos.

Esta capa es solo "presentación + llamada al modelo":
no entrena aquí; carga los artefactos generados por `src/entrenar.py`.
Los comentarios están pensados para estudiantes que ven despliegue por primera vez.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permite importar el paquete `src` cuando ejecutas: streamlit run app/streamlit_app.py
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import prediccion  # noqa: E402


def configurar_pagina() -> None:
    """Título de pestaña del navegador y layout amplio."""
    st.set_page_config(
        page_title="BancoDeTodos — Score demo",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_resource
def obtener_modelo_y_umbrales():
    """
    Carga modelo y umbrales una sola vez por proceso de Streamlit.

    @st.cache_resource evita releer el disco en cada interacción del usuario;
    en producción real usaríamos patrones similares (singleton, contenedor, etc.).
    """
    modelo = prediccion.cargar_pipeline_modelo()
    umbrales = prediccion.cargar_umbrales()
    return modelo, umbrales


def barra_color_banda(clave: str) -> str:
    """Devuelve un color HTML simple según la banda (solo para la UI)."""
    if clave == "bajo":
        return "#1a7f37"
    if clave == "medio":
        return "#b08900"
    return "#c62828"


def main() -> None:
    configurar_pagina()

    st.title("🏦 BancoDeTodos — Laboratorio de score de riesgo (demo)")
    st.markdown(
        """
        Esta aplicación **no usa datos reales** ni constituye asesoría financiera.
        Es un ejercicio académico: usted ingresa valores y el sistema muestra una
        **probabilidad estimada de incumplimiento** y una **banda de riesgo** definida
        por umbrales de negocio (archivo `artifacts/umbrales_riesgo.json`).
        """
    )

    with st.sidebar:
        st.header("Mapa del laboratorio MLOps")
        st.markdown(
            """
            1. **Datos sintéticos**: `data/creditos_sinteticos.csv`  
            2. **Entrenamiento**: `python -m src.entrenar` → guarda en `artifacts/`  
            3. **Inferencia**: módulo `src/prediccion.py`  
            4. **Esta app**: solo consume el modelo guardado (como en "producción").
            """
        )
        st.divider()
        st.caption("Si cambia el modelo, vuelva a ejecutar el entrenamiento y recargue la app.")

    modelo, umbrales = obtener_modelo_y_umbrales()

    col_form, col_info = st.columns([1, 1])

    with col_form:
        st.subheader("Datos del solicitante (ficticio)")
        st.caption("Ajuste los controles y pulse **Calcular score**.")

        ingreso = st.number_input(
            "Ingreso mensual (unidades del ejercicio)",
            min_value=0.0,
            value=3_000_000.0,
            step=50_000.0,
            help="Valor positivo; en el CSV sintético están en escala similar.",
        )
        deuda = st.number_input(
            "Deuda total",
            min_value=0.0,
            value=1_500_000.0,
            step=50_000.0,
        )
        antiguedad = st.number_input(
            "Antigüedad laboral (meses)",
            min_value=0,
            value=24,
            step=1,
        )
        num_creditos = st.number_input(
            "Número de créditos activos",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
        )
        moras = st.number_input(
            "Moras en los últimos 12 meses",
            min_value=0,
            max_value=12,
            value=0,
            step=1,
        )

        calcular = st.button("Calcular score", type="primary")

    with col_info:
        st.subheader("Resultado")
        if not calcular:
            st.info("Pulse **Calcular score** para ver la probabilidad y la banda.")
        else:
            fila = {
                "ingreso_mensual": ingreso,
                "deuda_total": deuda,
                "antiguedad_laboral_meses": int(antiguedad),
                "num_creditos_activos": int(num_creditos),
                "moras_ult_12m": int(moras),
            }
            try:
                resultado = prediccion.predecir_fila(fila, modelo=modelo, umbrales=umbrales)
            except Exception as exc:  # noqa: BLE001 — en clase queremos ver el error en pantalla
                st.error(f"Error al predecir: {exc}")
                return

            proba = resultado["probabilidad_incumplimiento"]
            clave = resultado["banda_clave"]
            texto = resultado["banda_texto"]
            color = barra_color_banda(clave)

            st.metric(
                label="Probabilidad de incumplimiento (clase positiva)",
                value=f"{proba:.1%}",
            )
            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background:{color}22;"
                f"border-left:6px solid {color};font-size:1.1rem;'><strong>{texto}</strong></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Umbrales: bajo si la probabilidad es menor que {umbrales['bajo_hasta']:.0%}; "
                f"medio si es menor que {umbrales['medio_hasta']:.0%}; "
                f"alto en cualquier otro caso."
            )

            with st.expander("Ver entradas enviadas al modelo"):
                st.json(resultado["entradas_utilizadas"])

    st.divider()
    st.subheader("¿Qué aprendemos con esto?")
    st.markdown(
        """
        - **Separación de roles**: el notebook o `src/entrenar.py` producen artefactos;
          la app solo los **sirve**.
        - **Contrato de datos**: el orden y nombres de columnas deben coincidir con el entrenamiento
          (`artifacts/features.json`).
        - **Decisión de negocio**: las bandas no las inventa el algoritmo; están en un archivo
          que el área de negocio podría cambiar sin reentrenar (en la práctica se coordina con riesgos).
        """
    )


if __name__ == "__main__":
    main()
