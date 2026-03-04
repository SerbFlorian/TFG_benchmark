import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.ticker import MultipleLocator

# --- CONFIGURACIÓN ---
# Directorio base relativo al propio script (independiente del CWD)
_BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = _BASE / 'data'
GRAFICAS_DIR = _BASE / 'graficas'
OUT_DIR      = GRAFICAS_DIR / 'sensibilidad_contexto'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_FILE = DATA_DIR / 'Benchmark_LongContext_Sensibilidad.xlsx'
OUTPUT_DPI = 300

# Precios oficiales de Gemini (por 1 millón de tokens)
PRECIO_INPUT_1M = 0.41706
PRECIO_OUTPUT_1M = 2.68168

# Constantes del tamaño del PDF (Tokens aproximados a sumar al Input base)
PDF_TOKENS = {
    10: 4700,
    50: 23500,
    100: 47000,
    200: 94000
}

# --- CONFIGURACIÓN DE RAG (Valores reales para 20 preguntas) ---
RAG_P_COST_TOTAL = 0.1313   # Suma Input (0.0186) + Output (0.0157) + Embedding (0.097)
RAG_N_COST_TOTAL = 0.2842   # Suma Input (0.2518) + Output (0.0324)
RAG_P_TIME_TOTAL = 9.52     # Segundos para 20 preguntas
RAG_N_TIME_TOTAL = 25.28    # Segundos para 20 preguntas
RAG_P_SCORE_AVG  = 6.8      # Nota media RAG Pinecone
RAG_N_SCORE_AVG  = 6.4      # Nota media RAG Native

# Valores por pregunta (Promedios unitarios)
RAG_P_COST_PER_Q = RAG_P_COST_TOTAL / 20
RAG_N_COST_PER_Q = RAG_N_COST_TOTAL / 20
RAG_P_LATENCY_PER_Q = 9.52    # segundos promedio por pregunta
RAG_N_LATENCY_PER_Q = 25.28   # segundos promedio por pregunta

# --- FUNCIONES AUXILIARES ---
def parsear_fraccion(valor):
    if pd.isna(valor): return 0
    valor_str = str(valor).strip()
    try: return float(valor_str)
    except: pass
    return 0

# --- CONFIGURACIÓN DE RAG ---
# Volvemos a tu valor original que conecta con el capítulo anterior del TFG
RAG_P_ROI = 51.8 
RAG_N_ROI = 22.5 

def procesar_datos():
    """Lee el Excel y calcula las medias (Coste, Latencia, Score) por cada volumen de páginas."""
    volumenes = [10, 50, 100, 200]
    resultados = {'paginas': [], 'coste': [], 'latencia': [], 'score': [], 'roi': []}
    
    for paginas in volumenes:
        sheet_name = f'{paginas}_Paginas'
        try:
            df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
            
            # Limpieza y conversión
            df['Time_seconds'] = pd.to_numeric(df['Time_seconds'], errors='coerce').fillna(0)
            df['Input_Tokens'] = pd.to_numeric(df['Input_Tokens'], errors='coerce').fillna(0)
            df['Output_Tokens'] = pd.to_numeric(df['Output_Tokens'], errors='coerce').fillna(0)
            df['Score'] = df['Question_score_Judge(0/10)'].apply(parsear_fraccion) if 'Question_score_Judge(0/10)' in df.columns else 0
            
            # Cálculo de tokens (Batch de 10)
            tokens_fijos_total = PDF_TOKENS[paginas] * 10
            sum_input = df['Input_Tokens'].head(10).sum() + tokens_fijos_total
            sum_output = df['Output_Tokens'].head(10).sum()
            
            # Costes del batch de 10
            cost_input_batch = (sum_input / 1_000_000) * PRECIO_INPUT_1M
            cost_output_batch = (sum_output / 1_000_000) * PRECIO_OUTPUT_1M
            cost_total_batch = cost_input_batch + cost_output_batch
            
            # Medias unitarias (Para las gráficas 1 y 2)
            media_coste = cost_total_batch / 10
            media_latencia = df['Time_seconds'].head(10).mean()
            media_score = df['Score'].head(10).mean()
            
            # CORRECCIÓN PARA EL ROI: 
            # Proyectamos el coste unitario a 20 preguntas para igualar la escala (0.1313€) del RAG
            coste_proyectado_20 = media_coste * 20
            media_roi = media_score / coste_proyectado_20 if coste_proyectado_20 > 0 else 0
            
            resultados['paginas'].append(paginas)
            resultados['coste'].append(media_coste)
            resultados['latencia'].append(media_latencia)
            resultados['score'].append(media_score)
            resultados['roi'].append(media_roi)
            
        except Exception as e:
            print(f"Advertencia: No se pudo procesar {sheet_name}. Error: {e}")
            
    return resultados

# --- GRÁFICA 0: CALIDAD CON PROMEDIO ---
def plot_calidad_promedio(datos):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mako_cmap = sns.color_palette('mako', as_cmap=True)
    color_p = mako_cmap(0.15)
    color_n = mako_cmap(0.5)
    color_lc = mako_cmap(0.50)

    scores = np.array(datos['score'])
    paginas = np.array(datos['paginas'])
    avg_total = scores.mean()
    
    ax.plot(paginas, scores, marker='o', markersize=10, linewidth=4, color=color_lc, label='Calidad Long Context')
    
    # Baselines de RAG
    ax.axhline(y=RAG_P_SCORE_AVG, color=color_p, linestyle='--', linewidth=2, label=f'RAG Pinecone: {RAG_P_SCORE_AVG:.1f}')
    ax.axhline(y=RAG_N_SCORE_AVG, color=color_n, linestyle=':', linewidth=2, label=f'RAG Native: {RAG_N_SCORE_AVG:.1f}')
    
    # Promedio Global
    ax.axhline(y=avg_total, color='#e74c3c', linestyle='-', alpha=0.5, linewidth=1.5, label=f'Promedio LC: {avg_total:.2f}')
    
    # Etiquetas de datos
    for x, y in zip(paginas, scores):
        ax.text(x, y + 0.3, f'{y:.2f}', color=color_lc, fontweight='bold', ha='center')

    ax.set_xlabel('Volumen de Contexto (Páginas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puntuación Media (Score 0-10)', fontsize=12, fontweight='bold')
    ax.set_title('Resolución de Calidad: Evolución vs Volumen', 
                 fontsize=15, fontweight='bold', loc='right', pad=30)
    ax.set_xticks(paginas)
    
    # ZOOM: Ajustamos el límite para que las líneas no se vean tan juntas
    ax.set_ylim(5.5, 8.5) 
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # ANOTACIONES DIRECTAS en las líneas horizontales (lado derecho)
    # x_pos = 205 para que esté justo después del último punto de 200
    x_pos = 202
    ax.text(x_pos, RAG_P_SCORE_AVG, ' ', color=color_p, va='center', fontweight='bold', fontsize=9)
    ax.text(x_pos, RAG_N_SCORE_AVG, ' ', color=color_n, va='center', fontweight='bold', fontsize=9)
    ax.text(x_pos, avg_total, ' ', color='#e74c3c', va='center', fontweight='bold', fontsize=9)

    # LEYENDA ARRIBA A LA IZQUIERDA FUERA
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), 
              fontsize=10, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'Tradeoff_0_Calidad.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print("   Gráfica 0 (Calidad) generada.")


# --- GRÁFICA 1: TRADE-OFF ECONÓMICO ---
def plot_tradeoff_economico(datos):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    costes = np.array(datos['coste'])
    paginas = np.array(datos['paginas'])
    
    mako_cmap = sns.color_palette('mako', as_cmap=True)
    color_p = mako_cmap(0.15)  # Pinecone
    color_n = mako_cmap(0.5)   # Native
    color_lc = mako_cmap(0.85) # Long Context

    # --- SOLUCIÓN AL PROBLEMA VISUAL DE SOMBREADO ---
    paginas_dense = np.linspace(paginas.min(), paginas.max(), 1000)
    costes_dense = np.interp(paginas_dense, paginas, costes)

    # 1. Calcular puntos de cruce exactos mediante interpolación
    x_cruce_pinecone = np.interp(RAG_P_COST_PER_Q, costes, paginas)
    x_cruce_nativo = np.interp(RAG_N_COST_PER_Q, costes, paginas)

    # 2. Etiquetas numéricas en los puntos de cruce
    ax.text(x_cruce_pinecone, RAG_P_COST_PER_Q, f' {int(x_cruce_pinecone)}', 
            color=color_p, fontweight='bold', va='bottom', ha='left', fontsize=11)
    
    ax.text(x_cruce_nativo, RAG_N_COST_PER_Q, f' {int(x_cruce_nativo)}', 
            color=color_n, fontweight='bold', va='bottom', ha='left', fontsize=11)

    # RAG Pinecone: línea discontinua (--)
    ax.axhline(y=RAG_P_COST_PER_Q, color=color_p, linestyle='--', linewidth=2.5,
               label=f'RAG Pinecone: {RAG_P_COST_PER_Q:.5f} €/q', zorder=4)
    
    # RAG Nativo: línea de puntos (:)
    ax.axhline(y=RAG_N_COST_PER_Q, color=color_n, linestyle=':', linewidth=2.5,
               label=f'RAG Nativo: {RAG_N_COST_PER_Q:.5f} €/q', zorder=4)
    
    # --- ZONAS DE COLOR USANDO ALTA RESOLUCIÓN ---
    # 1. Zona Verde: LC es más barato que Pinecone
    ax.fill_between(paginas_dense, costes_dense, RAG_P_COST_PER_Q,
                    where=(costes_dense <= RAG_P_COST_PER_Q), 
                    color='green', alpha=0.12, label='LC más barato que ambos RAG')
    
    # 2. Zona Azul Claro: LC entre Pinecone y Nativo
    ax.fill_between(paginas_dense, costes_dense, RAG_P_COST_PER_Q,
                    where=(costes_dense > RAG_P_COST_PER_Q) & (costes_dense <= RAG_N_COST_PER_Q), 
                    color='skyblue', alpha=0.3, label='LC más caro que Pinecone')

    # 3. Zona Roja: LC supera a RAG Nativo
    ax.fill_between(paginas_dense, costes_dense, RAG_P_COST_PER_Q,
                    where=(costes_dense > RAG_N_COST_PER_Q), 
                    color='red', alpha=0.1, label='LC más caro que RAG Pinecone')

    ax.set_xlabel('Volumen de Contexto (Páginas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coste Promedio por Pregunta (€)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off Económico: Break-even Point',
                 fontsize=14, fontweight='bold', loc='center', pad=15)
    ax.set_xticks(paginas)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    # Mover leyenda fuera de la gráfica, arriba a la izquierda
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, .99), 
              fontsize=9, frameon=True, shadow=True, ncol=1)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'Tradeoff_1_Economico.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print("   Gráfica 1 (Costes) generada.")


# --- GRÁFICA 2: TRADE-OFF DE LATENCIA ---
def plot_tradeoff_latencia(datos):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mako_cmap = sns.color_palette('mako', as_cmap=True)
    color_p = mako_cmap(0.15)
    color_n = mako_cmap(0.5)
    color_lc = mako_cmap(0.5)

    # Dibujar la línea de Long Context
    ax.plot(datos['paginas'], datos['latencia'], marker='s', markersize=8, linewidth=3, color=color_lc, label='Long Context (Tiempo Variable)')
    
    # Añadir los valores exactos encima de cada punto de Long Context
    for x, y in zip(datos['paginas'], datos['latencia']):
        ax.text(x, y + 0.6, f'{y:.2f}s', color=color_lc, fontweight='bold', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # RAG Pinecone: línea discontinua (--)
    ax.axhline(y=RAG_P_LATENCY_PER_Q, color=color_p, linestyle='--', linewidth=2.5, 
               label=f'RAG Pinecone (Fijo: {RAG_P_LATENCY_PER_Q:.2f}s)')
    
    # RAG Nativo: línea de puntos (:)
    ax.axhline(y=RAG_N_LATENCY_PER_Q, color=color_n, linestyle=':', linewidth=2.5, 
               label=f'RAG Nativo (Fijo: {RAG_N_LATENCY_PER_Q:.2f}s)')
    
    ax.set_xlabel('Volumen de Contexto (Páginas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo Promedio de Respuesta (Segundos)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off de Latencia: Velocidad vs Volumen', 
                 fontsize=15, fontweight='bold', loc='right', pad=25)
    ax.set_xticks(datos['paginas'])
    
    # Ajustar un poco el límite superior del eje Y para que los textos respiren bien
    ax.set_ylim(bottom=None, top=max(RAG_N_LATENCY_PER_Q, max(datos['latencia'])) + 3)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- LEYENDA FUERA DE LA GRÁFICA, ARRIBA A LA IZQUIERDA ---
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), 
              fontsize=10, frameon=True, shadow=True)

    plt.tight_layout()
    # bbox_inches='tight' evita que la caja externa sea recortada al guardar
    plt.savefig(OUT_DIR / 'Tradeoff_2_Latencia.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print("   Gráfica 2 (Latencia) generada con leyenda exterior.")

# --- GRÁFICA 3: TRADE-OFF DE EFICIENCIA (ROI) ---
def plot_tradeoff_eficiencia(datos):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mako_cmap = sns.color_palette('mako', as_cmap=True)
    color_p = mako_cmap(0.15)
    color_n = mako_cmap(0.5)
    color_lc = mako_cmap(0.50)

    # Eje Y en escala LINEAL para ROI (Seguimiento de la petición del usuario)
    ax.plot(datos['paginas'], datos['roi'], marker='D', markersize=8, linewidth=3, color=color_lc, label='Long Context (ROI)')
    ax.axhline(y=RAG_P_ROI, color=color_p, linestyle='--', linewidth=2.5, label=f'RAG Pinecone (ROI: {RAG_P_ROI:.1f})')
    ax.axhline(y=RAG_N_ROI, color=color_n, linestyle=':', linewidth=2.5, label=f'RAG Native (ROI: {RAG_N_ROI:.1f})')
    
    # Añadir los valores en cada punto
    for x, y in zip(datos['paginas'], datos['roi']):
        ax.text(x, y + (max(datos['roi']) * 0.03), f'{y:.1f}', color=color_lc, fontweight='bold', 
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel('Volumen de Contexto (Páginas)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eficiencia (Avg Score / Total Cost)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off de Eficiencia (ROI): Degrado por Volumen', 
                 fontsize=15, fontweight='bold', loc='right', pad=30)
    ax.set_xticks(datos['paginas'])
    
    # Ajustar límite Y para que los números no se corten arriba
    ax.set_ylim(bottom=0, top=max(datos['roi']) * 1.15)
    
    ax.grid(True, ls="--", alpha=0.4)
    # LEYENDA ARRIBA A LA IZQUIERDA FUERA
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), 
              fontsize=10, frameon=True, shadow=True, ncol=1)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'Tradeoff_3_Eficiencia.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print("  Gráfica 3 (Eficiencia/ROI) generada.")



# --- GRÁFICA 4: COSTE EXACTO POR HOJA ---
def plot_coste_exacto(datos):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mako_cmap = sns.color_palette('mako', n_colors=4) # Colores discretos para las barras
    
    # Calculamos el coste del lote de 10 acumulado (no la media unitaria)
    # media_coste = cost_total_batch / 10 -> cost_total_batch = media_coste * 10
    costes_lote = np.array(datos['coste']) * 10
    paginas = [f'{p} Págs' for p in datos['paginas']]
    
    bars = ax.bar(paginas, costes_lote, color=mako_cmap, alpha=0.8)
    
    # Etiquetas exactas en €
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(costes_lote)*0.01),
                f'{height:.4f}€', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Contexto Analizado', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coste Total del Lote (10 preguntas) (€)', fontsize=12, fontweight='bold')
    ax.set_title('Coste Exacto por Volumen de Información', 
                 fontsize=15, fontweight='bold', loc='center', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'Tradeoff_4_Coste_Exacto.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print("   Gráfica 4 (Coste Exacto) generada.")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    print("Iniciando análisis de Trade-offs (Sensibilidad al Contexto)...")
    datos_procesados = procesar_datos()
    
    if len(datos_procesados['paginas']) > 0:
        plot_calidad_promedio(datos_procesados)
        plot_tradeoff_economico(datos_procesados)
        plot_tradeoff_latencia(datos_procesados)
        plot_tradeoff_eficiencia(datos_procesados)
        plot_coste_exacto(datos_procesados)
        print("\n¡Todas las gráficas de Trade-off generadas con éxito!")
    else:
        print("Error: No se encontraron datos para generar las gráficas.")
