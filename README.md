# TFG Benchmark: Evaluación de Arquitecturas LLM

Este proyecto permite evaluar y comparar el rendimiento de diferentes arquitecturas de sistemas basados en modelos de lenguaje (LLM). Utiliza gpt-4o como juez automático para medir la precisión técnica y genera visualizaciones avanzadas para analizar métricas clave como calidad, latencia y costes.

## Estructura del Proyecto

El flujo de trabajo se divide en dos scripts principales:

### 1. Evaluación Automática (`llm_judge.py`)
Este script actúa como el "cerebro" de la evaluación. Procesa un archivo Excel con preguntas, respuestas de referencia (Ground Truth) y respuestas generadas por diferentes arquitecturas.

- **Funcionalidad**: Envía cada respuesta a GPT-4o con una rúbrica estricta (0-10) para obtener una puntuación y un razonamiento técnico.
- **Salida**: Actualiza el archivo Excel original (`TFG_Benchmark_Questions.xlsx`) añadiendo columnas con la puntuación y el razonamiento del "Juez LLM".
- **Configuración**: Requiere un archivo `.env.local` con una `OPENAI_API_KEY` válida.

### 2. Análisis y Visualización (`generar_graficas.py`)
Una vez obtenidos los resultados (ya sean humanos o del juez automático), este script genera un conjunto completo de gráficos para el análisis comparativo.

- **Visualizaciones incluidas**:
  - **Latencia**: Comparativa de tiempos de respuesta por arquitectura.
  - **Mapas de Calor (Heatmaps)**: Calidad detallada por pregunta.
  - **ROI (Coste vs Calidad)**: Análisis de eficiencia para identificar la mejor relación calidad-precio.
  - **Calidad + Contexto**: Gráficos de barras combinados con indicadores de éxito en la recuperación de información.
  - **Radar Global**: Un análisis 360° que normaliza Calidad, Eficiencia, Velocidad y Fiabilidad.
  - **Costes**: Desglose detallado (Tokens Input/Output y Embeddings) en escala logarítmica.
- **Versatilidad**: El script genera automáticamente dos versiones de los gráficos: una basada en la evaluación humana y otra en la del Juez LLM.

## Requisitos y Configuración

1. **Python 3.x** con las librerías: `pandas`, `openai`, `matplotlib`, `seaborn`, `numpy`, `python-dotenv` y `openpyxl`.
2. **Clave de API de OpenAI**: Configurada en un archivo `.env.local`.
3. **Datos**: Un archivo `TFG_Benchmark_Questions.xlsx` con las hojas correspondientes a cada arquitectura (`Arch1`, `Arch2`, `Arch3`).

## Uso

1. **Evaluar**: Ejecuta `python llm_judge.py` para que el juez analice las respuestas.
2. **Visualizar**: Ejecuta `python generar_graficas.py` para generar todas las imágenes de análisis en la carpeta raíz.

---
*Este proyecto ha sido desarrollado como parte de un Trabajo de Fin de Grado (TFG) centrado en el análisis de rendimiento de sistemas RAG y arquitecturas LLM.*
