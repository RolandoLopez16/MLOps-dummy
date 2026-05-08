# Laboratorio MLOps — BancoDeTodos (versión local, sin Docker)

Proyecto **didáctico** para una especialización en analítica: recorre un flujo **miniatura** de MLOps, desde datos sintéticos y notebook hasta una **aplicación Streamlit** que expone un **score de riesgo** (probabilidad de incumplimiento + bandas).

> **Aviso:** los datos son **100 % sintéticos**. No es un producto financiero ni constituye asesoría. El nombre **BancoDeTodos** es ficticio.

## Qué aprenderá el estudiante

1. **Generar datos** reproducibles y documentarlos.
2. **Explorar y comparar modelos** en Jupyter.
3. **Entrenar de forma reproducible** con un script (`src/entrenar.py`) que guarda **artefactos**.
4. **Separar entrenamiento de inferencia**: la app solo **carga** el modelo guardado (como en un entorno de producción simplificado).
5. **Definir reglas de negocio** (bandas de riesgo) aparte del algoritmo (`artifacts/umbrales_riesgo.json`).

## Estructura del repositorio

| Ruta | Rol |
|------|-----|
| `data/creditos_sinteticos.csv` | Dataset sintético listo para entrenar. |
| `data/diccionario_datos.json` | Descripción de columnas (metadatos). |
| `notebooks/01_exploracion_y_modelo.ipynb` | Exploración y comparación de modelos en clase. |
| `src/generar_datos.py` | Vuelve a crear el CSV si quiere regenerar datos. |
| `src/entrenar.py` | Entrena y escribe artefactos en `artifacts/`. |
| `src/prediccion.py` | Funciones para cargar el modelo y predecir. |
| `app/streamlit_app.py` | Interfaz web **BancoDeTodos**. |
| `artifacts/` | Modelo entrenado (`modelo_pipeline.joblib`), umbrales y lista de variables. |

Los artefactos **vienen incluidos** para que pueda ejecutar la app sin entrenar primero. Si cambia datos o código de entrenamiento, debe **volver a entrenar** para actualizarlos.

## Requisitos

- **Python 3.10 o superior** (recomendado 3.11).
- **pip** (gestor de paquetes).
- Navegador web (Chrome, Edge, Firefox, etc.).

No se requiere Docker en esta versión del laboratorio.

## Instalación paso a paso (Windows / macOS / Linux)

Abra una terminal en la **carpeta raíz** del proyecto (donde está este `README.md`).

### 1. Crear un entorno virtual (recomendado)

Un entorno virtual evita mezclar librerías con otros proyectos.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. (Opcional) Regenerar datos sintéticos

Solo si desea un nuevo CSV con la misma lógica y semilla fija en el código:

```bash
python -m src.generar_datos
```

### 4. (Opcional) Volver a entrenar el modelo

Después de cambiar el CSV o el código en `src/entrenar.py`:

```bash
python -m src.entrenar
```

Verá en consola métricas básicas (AUC y accuracy) **solo con fines ilustrativos**.

### 5. Ejecutar la aplicación Streamlit

```bash
streamlit run app/streamlit_app.py
```

El navegador abrirá una URL local (por defecto `http://localhost:8501`). Allí puede ingresar valores y ver **probabilidad de incumplimiento** y **banda de riesgo**.

Para detener el servidor, use `Ctrl+C` en la terminal.

---

## Tutorial: recorra el laboratorio en orden (guía del profesor para usted)

Bienvenido. En esta sección **no** solo listamos archivos: vamos a **recorrer el flujo** como lo haríamos en clase, del concepto a la “producción” de juguete. Le recomiendo abrir cada archivo en el editor mientras lee el paso correspondiente.

### La idea que debe quedar clara

> **El notebook sirve para experimentar.** Los **scripts** y la carpeta **`artifacts/`** sirven para **reproducir** lo que decidimos. **Streamlit** no entrena: **solo consume** el modelo ya guardado, como suele ocurrir en un despliegue real (la app y el entrenamiento viven en procesos distintos).

Si puede repetir esa frase con sus propias palabras, ya entendió el corazón del laboratorio.

### Paso 0 — Orientación

**Archivo:** este `README.md`.

**Qué hace:** le da el mapa del repositorio, los comandos y el aviso legal/didáctico. Vuelva aquí cuando se pierda.

### Paso 1 — El problema y los datos (aún sin modelos)

**Archivos (ábralos en este orden):**

1. `data/diccionario_datos.json` — lea qué significa cada columna. Es el **diccionario de datos** del ejercicio.
2. `data/creditos_sinteticos.csv` — observe las primeras filas (desde el editor o Excel). La columna **`incumplimiento`** es lo que el modelo intentará predecir.

**Qué debe entender:** aquí definimos el **contrato de datos**. Si mañana cambian los nombres o el significado de las columnas, el entrenamiento y la app deben actualizarse de forma coherente.

### Paso 2 — Exploración y comparación de modelos

**Archivo:** `notebooks/01_exploracion_y_modelo.ipynb`.

**Qué hace:** carga el CSV, muestra estadísticas, divide entrenamiento/prueba y compara al menos dos enfoques (por ejemplo, regresión logística y un bosque aleatorio).

**Qué debe entender:** el cuaderno es el lugar de **preguntas abiertas** y pruebas rápidas. En la industria también se usa Jupyter, pero **no** se “sube el notebook” tal cual como sistema productivo: se traduce lo aprendido a **scripts** y **versiones** de artefactos.

**Cómo ejecutarlo:** instale dependencias, luego `jupyter notebook` desde la raíz del proyecto y abra el cuaderno. Si tiene dudas con el kernel, vea la sección **Uso del notebook** más abajo en este mismo documento.

### Paso 3 — ¿De dónde salen los datos sintéticos? (opcional pero muy formativo)

**Archivo:** `src/generar_datos.py`.

**Qué hace:** genera de nuevo `data/creditos_sinteticos.csv` con una **semilla fija** (reproducibilidad) y reglas documentadas en comentarios.

**Comando (si lo prueba):**

```bash
python -m src.generar_datos
```

**Qué debe entender:** en un caso real los datos vendrían de otro lado; aquí **simulamos** el mundo para que usted practique el flujo sin datos personales. Si regenera el CSV, recuerde **volver a entrenar** (`python -m src.entrenar`) antes de confiar en las predicciones de la app.

### Paso 4 — Entrenamiento reproducible y artefactos

**Archivo:** `src/entrenar.py`.

**Qué hace:** lee el CSV, entrena un **pipeline** de scikit-learn (escalado + modelo), imprime métricas sencillas y **escribe** los archivos en `artifacts/`.

**Comando:**

```bash
python -m src.entrenar
```

**Archivos que debe reconocer en `artifacts/` después de entrenar:**

| Archivo | Para qué sirve |
|---------|----------------|
| `modelo_pipeline.joblib` | El modelo ya entrenado listo para cargar (no es texto legible: es binario). |
| `features.json` | Lista y **orden** de las variables que el modelo exige al predecir. |
| `umbrales_riesgo.json` | Cortes para pasar de probabilidad a **bajo / medio / alto** (decisión de negocio). |

**Qué debe entender:** este script es el análogo de “**entrenamiento en pipeline**”: mismo código, mismos datos (versionados), mismos artefactos. Eso es lo que luego se automatiza con CI/CD en empresas.

### Paso 5 — Inferencia: separar “entrenar” de “predecir”

**Archivo:** `src/prediccion.py`.

**Qué hace:** carga el `.joblib` y los JSON, arma la entrada como el modelo la espera y devuelve probabilidad + banda.

**Qué debe entender:** la **inferencia** es un contrato: entradas válidas → salida. La aplicación web **no** debe duplicar esta lógica a mano; debe **importarla** o llamar a un servicio que la encapsule. Aquí lo hacemos con importaciones de Python para mantener el ejemplo corto.

### Paso 6 — La cara al usuario: Streamlit como “despliegue” de demostración

**Archivo:** `app/streamlit_app.py`.

**Qué hace:** muestra el formulario **BancoDeTodos**, llama a las funciones de `prediccion.py` y enseña la probabilidad y el color de la banda. Usa caché para no recargar el modelo en cada clic.

**Comando:**

```bash
streamlit run app/streamlit_app.py
```

**Qué debe entender:** en producción real habría autenticación, monitoreo, escalado, etc. Aquí solo ilustramos el **último eslabón**: servir predicciones a partir de artefactos versionados.

### Paso 7 — Tabla resumen (léala antes del examen o la defensa)

| Orden | Archivo o carpeta | Etapa MLOps |
|------:|-------------------|-------------|
| 0 | `README.md` | Orientación |
| 1 | `data/` + `diccionario_datos.json` | Datos y significado |
| 2 | `notebooks/01_exploracion_y_modelo.ipynb` | Experimentación |
| 3 | `src/generar_datos.py` | Origen sintético (opcional) |
| 4 | `src/entrenar.py` → `artifacts/` | Entrenamiento reproducible |
| 5 | `src/prediccion.py` | Lógica de inferencia |
| 6 | `app/streamlit_app.py` | Interfaz / despliegue local |

### Actividades que le sugiero probar

1. **Cambiar negocio sin reentrenar:** edite `artifacts/umbrales_riesgo.json`, guarde, recargue la app en el navegador y observe cómo cambia la **banda** con la misma probabilidad del modelo.
2. **Coherencia notebook → script:** si en el cuaderno el bosque aleatorio le gusta más, intente (como debería) modificar `src/entrenar.py` para guardar ese modelo, vuelva a entrenar y compare en Streamlit.

---

## Uso del notebook

1. Instale dependencias (incluye Jupyter).
2. Desde la raíz del proyecto:

   ```bash
   jupyter notebook
   ```

3. Abra `notebooks/01_exploracion_y_modelo.ipynb`.

Si Jupyter no encuentra el kernel del entorno virtual, seleccione el kernel asociado a `.venv` o ejecute:

```bash
python -m ipykernel install --user --name=bancodetodos --display-name "Python (BancoDeTodos)"
```

## Bandas de riesgo (decisión de negocio)

El modelo entrega una **probabilidad** entre 0 y 1. Las **bandas** se definen en `artifacts/umbrales_riesgo.json`:

- **Riesgo bajo:** probabilidad menor que 30 %.
- **Riesgo medio:** entre 30 % y 60 % (sin incluir 60 % en “bajo”).
- **Riesgo alto:** 60 % o más.

Puede editar ese archivo para enseñar que **negocio** y **modelo** son decisiones distintas (siempre coordinando con buenas prácticas de riesgo en la vida real).

## Próxima fase (fuera de este README)

En una segunda entrega del curso se puede añadir **Docker** para empaquetar la misma app. Streamlit y Docker son compatibles; solo se agrega un `Dockerfile` y comandos de construcción/ejecución.

## Preguntas frecuentes

**¿Por qué hay un script si ya tengo el notebook?**  
El notebook sirve para explorar; el script garantiza **reproducibilidad** y es lo que se automatizaría en pipelines (CI/CD) en empresas.

**¿Puedo subir mi propio CSV?**  
Sí, si mantiene **exactamente** las columnas esperadas (`features.json`). De lo contrario, debe adaptar `src/entrenar.py` y la app.

**¿Los resultados son financieramente válidos?**  
No. Es un **dummy** para enseñar flujo técnico y conceptos de despliegue.

## Licencia educativa

Uso libre en contexto académico. Cite el repositorio si lo reutiliza en otro curso.
