import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

# --- CONFIGURACIÓN ---
# Archivo fuente de datos y resolución de salida de las imágenes
EXCEL_FILE = 'TFG_Benchmark_Questions.xlsx'
OUTPUT_DPI = 300

# --- FUNCIONES AUXILIARES DE PROCESAMIENTO ---
def parsear_fraccion(valor):
    """Convierte fracciones a números decimales."""
    if pd.isna(valor): return 0
    valor_str = str(valor).strip()
    try: return float(valor_str)
    except: pass
    if '/' in valor_str:
        try:
            partes = valor_str.split('/')
            return float(partes[0].strip())
        except: return 0
    return 0

def limpiar_columna_numerica(serie):
    """Convierte una serie a numérica, tratando errores como NaN y llenándolos con 0."""
    return pd.to_numeric(serie, errors='coerce').fillna(0)

def cargar_datos():
    """Carga las tres hojas del Excel y retorna DataFrames"""
    df1 = pd.read_excel(EXCEL_FILE, sheet_name='Arch1')
    df2 = pd.read_excel(EXCEL_FILE, sheet_name='Arch2')
    df3 = pd.read_excel(EXCEL_FILE, sheet_name='Arch3')
    
    for df, prefix in [(df1, 'Arch1'), (df2, 'Arch2'), (df3, 'Arch3')]:
        if f'{prefix}_Time_seconds' in df.columns:
            df[f'{prefix}_Time_seconds'] = limpiar_columna_numerica(df[f'{prefix}_Time_seconds'])
        
        # MODIFICACIÓN 1: Limpiar tanto la columna humana como la del Juez
        col_score_humano = f'{prefix}_Question_score_(0/10)'
        col_score_judge = f'{prefix}_Question_score_Judge(0/10)'
        
        if col_score_humano in df.columns:
            df[col_score_humano] = df[col_score_humano].apply(parsear_fraccion)
        if col_score_judge in df.columns:
            df[col_score_judge] = df[col_score_judge].apply(parsear_fraccion)
            
    return df1, df2, df3

def convertir_context_found_binario(valor):
    """Auxiliar para normalizar respuestas de 'Context Found' a formato binario (0 para Sí, 1 para No)."""
    if pd.isna(valor): return 0
    val_str = str(valor).strip().lower()
    return 0 if val_str in ['sí', 'si', 'yes', 's'] else 1

# --- GRÁFICA 1: LATENCIA PROMEDIO ---
# Representa el tiempo medio de respuesta ignorando la calidad
def grafica_latencia(df1, df2, df3):
    arquitecturas = ['Long Context', 'RAG Pinecone', 'RAG Native']
    valores = [
        df3['Arch3_Time_seconds'].mean(),
        df1['Arch1_Time_seconds'].mean(),
        df2['Arch2_Time_seconds'].mean(),
    ]
    mako_cmap = sns.color_palette('mako', as_cmap=True)
    colores = [mako_cmap(0.85), mako_cmap(0.15), mako_cmap(0.5)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(arquitecturas, valores, color=colores)

    for i, (bar, val) in enumerate(zip(bars, valores)):
        if i == 2:
            ax.text(val - 0.3, i, f'{val:.2f}s', va='center', ha='right', fontweight='bold', fontsize=11, color='white')
        else:
            ax.text(val + 0.2, i, f'{val:.2f}s', va='center', fontweight='bold', fontsize=11)

    mult = valores[2] / valores[0]
    ax.text(valores[2] + 0.35, 2, f'({mult:.1f}x más lento que Long Context)', va='center', fontsize=9.5, color='#c0392b', fontweight='bold')
    ax.set_xlim(0, max(valores) * 1.6)
    ax.set_xlabel('Tiempo Promedio (segundos)', fontsize=12, fontweight='bold')
    ax.set_title('Latencia Promedio por Arquitectura', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=5, color='#e74c3c', linestyle='--', linewidth=1.8, alpha=0.85)
    ax.text(5.1, 1.2, 'Umbral (5s)', color='#e74c3c', fontsize=9, va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Grafica_Latencia.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print("  Gráfica 1: Latencia guardada")

# --- GRÁFICA 2: CALIDAD (HEATMAP) ---
# Genera un mapa de calor con las puntuaciones de cada pregunta para las 3 arquitecturas
def grafica_calidad(df1, df2, df3, is_judge=False):
    col_suffix = "Question_score_Judge(0/10)" if is_judge else "Question_score_(0/10)"
    file_suffix = "Judge" if is_judge else "Humana"
    title_extra = "(LLM-Judge)" if is_judge else "(Evaluación Humana Experta)"
    
    num_preguntas = 20
    matriz = np.zeros((num_preguntas, 3))
    
    for i in range(num_preguntas):
        if i < len(df1): matriz[i, 0] = df1.iloc[i][f'Arch1_{col_suffix}']
        if i < len(df2): matriz[i, 1] = df2.iloc[i][f'Arch2_{col_suffix}']
        if i < len(df3): matriz[i, 2] = df3.iloc[i][f'Arch3_{col_suffix}']
    
    fig, ax = plt.subplots(figsize=(10, 14))
    im = sns.heatmap(matriz, annot=True, fmt='.0f', cmap='mako', vmin=0, vmax=10, 
                     cbar_kws={'label': 'Calidad de Respuesta (0-10)', 'shrink': 0.8},
                     linewidths=1.2, linecolor='white', square=False, 
                     annot_kws={'fontsize': 10, 'fontweight': 'bold'}, ax=ax)
    
    ax.set_xticklabels(['RAG Pinecone', 'RAG Native', 'Long Context'], fontsize=12, fontweight='bold')
    ax.set_yticklabels([f'P{i}' for i in range(1, num_preguntas + 1)], fontsize=10, rotation=0)

    ax.set_ylabel('Pregunta', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title(f'Calidad de Respuesta por Pregunta y Arquitectura\n{title_extra}', fontsize=15, fontweight='bold', pad=20)
    
    promedios = matriz.mean(axis=0)
    stats_text = f"CALIDAD PROMEDIO:\n\n"
    stats_text += f"RAG Pinecone: {np.ceil(promedios[0]*10)/10:.1f}/10\n"
    stats_text += f"RAG Native:   {np.ceil(promedios[1]*10)/10:.1f}/10\n"
    stats_text += f"Long Context: {np.ceil(promedios[2]*10)/10:.1f}/10"
    
    ax.text(1.5, -2, stats_text, fontsize=12, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#f8f9fa', alpha=0.95, edgecolor='#dee2e6', linewidth=1.5),
            ha='center', fontweight='bold', family='monospace') 
    
    plt.tight_layout()
    plt.savefig(f'Grafica_Calidad_{file_suffix}.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica 2: Calidad guardada como 'Grafica_Calidad_{file_suffix}.png'")

# --- GRÁFICA 3: ROI (COSTO VS CALIDAD) ---
# Gráfico de dispersión que muestra qué arquitectura ofrece mejor relación calidad-precio
def grafica_roi(df1, df2, df3, is_judge=False):
    col_suffix = "Question_score_Judge(0/10)" if is_judge else "Question_score_(0/10)"
    file_suffix = "Judge" if is_judge else "Humana"
    title_extra = "(LLM-Judge)" if is_judge else "(Humano)"

    PRECIO_INPUT_1M, PRECIO_OUTPUT_1M, COST_EMBEDDING_OPENAI = 0.41706, 2.68168, 0.097
    
    # Cálculo manual del coste estimado basado en tokens medidos previamente
    costes = {
        'RAG Pinecone': (44575 / 1_000_000) * PRECIO_INPUT_1M + (5845 / 1_000_000) * PRECIO_OUTPUT_1M + COST_EMBEDDING_OPENAI,
        'RAG Native': (603739 / 1_000_000) * PRECIO_INPUT_1M + (12067 / 1_000_000) * PRECIO_OUTPUT_1M,
        'Long Context': (2828444 / 1_000_000) * PRECIO_INPUT_1M + (4462 / 1_000_000) * PRECIO_OUTPUT_1M
    }
    
    quality_promedio = {
        'RAG Pinecone': np.ceil(df1[f'Arch1_{col_suffix}'].head(20).mean() * 10) / 10,
        'RAG Native': np.ceil(df2[f'Arch2_{col_suffix}'].head(20).mean() * 10) / 10,
        'Long Context': np.ceil(df3[f'Arch3_{col_suffix}'].head(20).mean() * 10) / 10
    }
    
    eficiencia = {k: quality_promedio[k] / costes[k] for k in costes.keys()}
    arquitecturas = list(costes.keys())
    x_values, y_values = [costes[k] for k in arquitecturas], [quality_promedio[k] for k in arquitecturas]
    colors = ['#440154', '#21918c', '#fde725']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, arch in enumerate(arquitecturas):
        ax.scatter(x_values[i], y_values[i], s=800, color=colors[i], edgecolors='black', linewidth=3, alpha=0.85, label=arch, zorder=3)
        xy_text_offset = (-20, 15) if arch == 'Long Context' else (15, 15)
        ha_align = 'right' if arch == 'Long Context' else 'left'
        ax.annotate(f'{arch}\nEficiencia: {eficiencia[arch]:.1f}', (x_values[i], y_values[i]), 
                    xytext=xy_text_offset, textcoords='offset points', fontsize=11, fontweight='bold', ha=ha_align,
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
    
    sorted_indices = np.argsort(x_values)
    x_sorted, y_sorted = [x_values[i] for i in sorted_indices], [y_values[i] for i in sorted_indices]
    ax.plot(x_sorted, y_sorted, 'k--', alpha=0.3, linewidth=2, zorder=1)
    
    ax.set_xlabel('Coste Total (€ EUR)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Calidad Promedio (Score 0-10)', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_title(f'Trade-off Coste vs Calidad {title_extra}\n(Arriba-Izquierda = Mejor: Alta Calidad, Bajo Coste)', fontsize=15, fontweight='bold', pad=20)
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='none', edgecolor='none', label='Frontera de Eficiencia:'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=12, label='  RAG Pinecone', markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=12, label='  RAG Native', markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=12, label='  Long Context', markeredgecolor='black', markeredgewidth=2),
    ]
    legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True, shadow=True, fancybox=True)
    legend.get_texts()[0].set_weight('bold')
    legend.get_texts()[0].set_fontsize(12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, max(x_values) * 1.15)
    ax.set_ylim(min(y_values) - 0.3, max(y_values) + 0.3)
    
    plt.tight_layout()
    plt.savefig(f'Grafica_ROI_{file_suffix}.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica 3: ROI guardada como 'Grafica_ROI_{file_suffix}.png'")

# --- GRÁFICA 4: CALIDAD + CONTEXTO COMBINADOS ---
# Compara puntuaciones (barras) con el éxito en encontrar contexto (iconos verde/rojo)
def grafica_calidad_contexto_combinada(df1, df2, df3, is_judge=False):
    col_suffix = "Question_score_Judge(0/10)" if is_judge else "Question_score_(0/10)"
    file_suffix = "Judge" if is_judge else "Humana"
    title_extra = "(LLM-Judge)" if is_judge else "(Evaluación Humana)"

    num_preguntas = 20
    arch1_scores = df1[f'Arch1_{col_suffix}'].head(num_preguntas).values
    arch2_scores = df2[f'Arch2_{col_suffix}'].head(num_preguntas).values
    arch3_scores = df3[f'Arch3_{col_suffix}'].head(num_preguntas).values
    
    arch1_context = df1['Arch1_Context_Found_(Sí/No)'].head(num_preguntas).values
    arch2_context = df2['Arch2_Context_Found_(Sí/No)'].head(num_preguntas).values
    arch3_context = df3['Arch3_Context_Found_(Sí/No)'].head(num_preguntas).values
    
    preguntas = [f'P{i}' for i in range(1, num_preguntas + 1)]
    fig, ax = plt.subplots(figsize=(14, 8))
    x, width = np.arange(len(preguntas)), 0.25
    
    for i in range(0, len(preguntas), 2):
        ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.07, zorder=0)
    
    mako_cmap = sns.color_palette('mako', as_cmap=True)
    color_arch1, color_arch2, color_arch3 = mako_cmap(0.15), mako_cmap(0.5), mako_cmap(0.85)
    
    bars1 = ax.bar(x - width, arch1_scores, width, label='RAG Pinecone', color=color_arch1, alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, arch2_scores, width, label='RAG Native', color=color_arch2, alpha=0.9, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, arch3_scores, width, label='Long Context', color=color_arch3, alpha=0.9, edgecolor='white', linewidth=0.5)
    
    for i, (c1, c2, c3) in enumerate(zip(arch1_context, arch2_context, arch3_context)):
        for j, (c, scores, offset) in enumerate([(c1, arch1_scores, -width), (c2, arch2_scores, 0), (c3, arch3_scores, width)]):
            c_str = str(c).strip().upper().replace('Í', 'I')
            marker, color = ('o', 'green') if c_str in ['SI', 'YES', 'S'] else ('x', 'red')
            if marker == 'o':
                ax.scatter(i + offset, scores[i] + 0.3, marker=marker, s=100, color=color, edgecolors='black', linewidths=1.5, zorder=3)
            else:
                ax.scatter(i + offset, scores[i] + 0.3, marker=marker, s=100, color=color, linewidths=2, zorder=3)
    
    ax.set_xlabel('Pregunta', fontsize=13, fontweight='bold')
    ax.set_ylabel('Calidad (Score 0-10)', fontsize=13, fontweight='bold')
    ax.set_title(f'Calidad de Respuesta + Contexto Encontrado\n{title_extra}', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(preguntas, rotation=45, ha='right')
    ax.set_ylim(0, 11.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    color_legend = [Patch(facecolor=color_arch1, edgecolor='black', label='RAG Pinecone'), Patch(facecolor=color_arch2, edgecolor='black', label='RAG Native'), Patch(facecolor=color_arch3, edgecolor='black', label='Long Context')]
    legend_colors = ax.legend(handles=color_legend, loc='upper left', fontsize=10, frameon=True, shadow=True, title='Arquitecturas', bbox_to_anchor=(0, 1.2))
    ax.add_artist(legend_colors) 
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Contexto Encontrado (Sí)', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Contexto NO Encontrado', markeredgecolor='red', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, shadow=True, title='Contexto', bbox_to_anchor=(1, 1.2))
    
    plt.tight_layout()
    plt.savefig(f'Grafica_Calidad_Contexto_{file_suffix}.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica 4: Calidad+Contexto guardada como 'Grafica_Calidad_Contexto_{file_suffix}.png'")

# --- GRÁFICA 5: COMPARATIVA GLOBAL (RADAR CHART AVANZADO) ---
# Normaliza y compara 4 métricas clave: Calidad, Eficiencia, Velocidad y Fiabilidad
def grafica_comparativa_arquitecturas(df1, df2, df3, is_judge=False):
    col_suffix = "Question_score_Judge(0/10)" if is_judge else "Question_score_(0/10)"
    file_suffix = "Judge" if is_judge else "Humana"
    title_extra = " (LLM-Judge)" if is_judge else " (Evaluación Humana)"

    PRECIO_INPUT_1M, PRECIO_OUTPUT_1M, COST_EMBEDDING_OPENAI = 0.41706, 2.68168, 0.097
    costes = {
        'Arch1': (44575 / 1_000_000) * PRECIO_INPUT_1M + (5845 / 1_000_000) * PRECIO_OUTPUT_1M + COST_EMBEDDING_OPENAI,
        'Arch2': (603739 / 1_000_000) * PRECIO_INPUT_1M + (12067 / 1_000_000) * PRECIO_OUTPUT_1M,
        'Arch3': (2828444 / 1_000_000) * PRECIO_INPUT_1M + (4462 / 1_000_000) * PRECIO_OUTPUT_1M
    }
    
    calidad = {
        'Arch1': np.ceil(df1[f'Arch1_{col_suffix}'].head(20).mean() * 10) / 10,
        'Arch2': np.ceil(df2[f'Arch2_{col_suffix}'].head(20).mean() * 10) / 10,
        'Arch3': np.ceil(df3[f'Arch3_{col_suffix}'].head(20).mean() * 10) / 10
    }
    
    latencia = {'Arch1': df1['Arch1_Time_seconds'].head(20).mean(), 'Arch2': df2['Arch2_Time_seconds'].head(20).mean(), 'Arch3': df3['Arch3_Time_seconds'].head(20).mean()}
    
    def calcular_tasa_contexto(df, col_name):
        return (df[col_name].head(20).apply(lambda x: 1 if str(x).strip().lower() in ['sí', 'si', 'yes', 's'] else 0).sum() / 20) * 100 
    
    contexto_encontrado = {'Arch1': calcular_tasa_contexto(df1, 'Arch1_Context_Found_(Sí/No)'), 'Arch2': calcular_tasa_contexto(df2, 'Arch2_Context_Found_(Sí/No)'), 'Arch3': calcular_tasa_contexto(df3, 'Arch3_Context_Found_(Sí/No)')}
    
    
    # NORMALIZACIÓN DE MÉTRICAS (Escala 0-10 para el Radar)
    calidad_norm = {k: v for k, v in calidad.items()}
    max_coste = max(costes.values())
    coste_norm = {k: 10 * (1 - v/max_coste) for k, v in costes.items()}
    max_latencia = max(latencia.values())
    velocidad_norm = {k: 10 * (1 - v/max_latencia) for k, v in latencia.items()}
    fiabilidad_norm = {k: v / 10 for k, v in contexto_encontrado.items()}
    
    categorias = ['Calidad\n(Score)', 'Eficiencia\n(Bajo Coste)', 'Velocidad\n(Baja Latencia)', 'Fiabilidad\n(Contexto OK)']
    valores_arch1 = [calidad_norm['Arch1'], coste_norm['Arch1'], velocidad_norm['Arch1'], fiabilidad_norm['Arch1']]
    valores_arch2 = [calidad_norm['Arch2'], coste_norm['Arch2'], velocidad_norm['Arch2'], fiabilidad_norm['Arch2']]
    valores_arch3 = [calidad_norm['Arch3'], coste_norm['Arch3'], velocidad_norm['Arch3'], fiabilidad_norm['Arch3']]
    
    valores_arch1 += valores_arch1[:1]
    valores_arch2 += valores_arch2[:1]
    valores_arch3 += valores_arch3[:1]
    angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
    angulos += angulos[:1]
    
    fig = plt.figure(figsize=(16, 15))
    gs = plt.GridSpec(4, 3, height_ratios=[4, 0.4, 1.8, 0.6], hspace=0.3)
    ax_main = fig.add_subplot(gs[0:2, :], projection='polar')
    
    nombres, colores = ['RAG Pinecone', 'RAG Native', 'Long Context'], ['#440154', '#21918c', '#fde725']
    valores_all = [valores_arch1, valores_arch2, valores_arch3]
    promedio_global = np.mean(valores_all, axis=0).tolist()
    
    ax_main.plot(angulos, promedio_global, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='Promedio Global', zorder=1)
    ax_main.fill(angulos, promedio_global, color='gray', alpha=0.02, zorder=1)

    for i, (valores, nombre, color) in enumerate(zip(valores_all, nombres, colores)):
        ax_main.fill(angulos, valores, alpha=0.05, color=color, zorder=2)
        ax_main.fill(angulos, valores, alpha=0.1, color=color, zorder=2)
        ax_main.plot(angulos, valores, linewidth=4, color=color, alpha=0.9, zorder=3)
        ax_main.plot(angulos, valores, linewidth=1, color='white', alpha=0.5, zorder=3)
        ax_main.scatter(angulos, valores, s=80, color=color, edgecolors='white', linewidth=1.5, zorder=4)

    for j, angle in enumerate(angulos[:-1]):
        puntos_eje = [{'val': valores_all[i][j], 'color': colores[i], 'type': 'arch'} for i in range(len(valores_all))]
        puntos_eje.append({'val': promedio_global[j], 'color': 'gray', 'type': 'avg'})
        puntos_eje.sort(key=lambda x: x['val'])
        
        last_radial_pos = -10
        for punto in puntos_eje:
            val, color = punto['val'], punto['color']
            radial_pos = max(val + 0.6, last_radial_pos + 0.8)
            last_radial_pos = radial_pos
            ha, va = 'center', 'center'
            if abs(angle - 0) < 0.1: ha, va = 'center', 'bottom'
            elif abs(angle - np.pi/2) < 0.1: ha, va = 'left', 'center'
            elif abs(angle - np.pi) < 0.1: ha, va = 'center', 'top'
            elif abs(angle - 3*np.pi/2) < 0.1: ha, va = 'right', 'center'
            
            if punto['type'] == 'avg':
                ax_main.text(angle, radial_pos, f'{val:.1f}', ha=ha, va=va, fontsize=8, fontweight='bold', color='#555555', zorder=10, bbox=dict(boxstyle='round,pad=0.15', facecolor='#f0f0f0', alpha=0.8, edgecolor='gray', linewidth=0.5))
            else:
                ax_main.text(angle, radial_pos, f'{val:.1f}', ha=ha, va=va, fontsize=9, fontweight='bold', color='white', zorder=10, bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.9, edgecolor='none'))

    ax_main.set_xticks(angulos[:-1])
    ax_main.set_xticklabels([], alpha=0)
    
    label_dist = 11.3
    for j, (angle, label) in enumerate(zip(angulos[:-1], categorias)):
        ha, va, curr_dist = 'center', 'center', label_dist
        if "Calidad" in label: final_text, ha, va, curr_dist = label.replace("\n", " "), 'center', 'bottom', label_dist + 0.5
        elif "Eficiencia" in label: final_text, ha, va, curr_dist = label, 'left', 'center', label_dist + 0.1
        elif "Fiabilidad" in label: final_text, ha, va, curr_dist = label, 'right', 'center', label_dist + 1
        elif "Velocidad" in label: final_text, ha, va, curr_dist = label.replace("\n", " "), 'center', 'bottom', label_dist + 0.3
        ax_main.text(angle, curr_dist, final_text, ha=ha, va=va, fontsize=12, fontweight='bold', color='black')

    ax_main.set_theta_offset(np.pi / 2)
    ax_main.set_theta_direction(-1)
    ax_main.set_ylim(0, 11)
    ax_main.set_yticks([2, 4, 6, 8, 10])
    ax_main.set_yticklabels([], alpha=0)
    ax_main.grid(True, linestyle=(0, (5, 10)), alpha=0.3, color='gray')
    ax_main.spines['polar'].set_visible(False)
    ax_main.set_title(f'ANÁLISIS COMPARATIVO GLOBAL{title_extra}\n(Métricas Escala 0-10 | Más afuera = Mejor)', fontsize=18, fontweight='bold', pad=45)

    for i, (valores, nombre, color) in enumerate(zip(valores_all, nombres, colores)):
        ax_sub = fig.add_subplot(gs[2, i], projection='polar')
        ax_sub.fill(angulos, promedio_global, color='gray', alpha=0.05)
        ax_sub.plot(angulos, valores, linewidth=2, color=color)
        ax_sub.fill(angulos, valores, alpha=0.3, color=color)
        ax_sub.set_xticks(angulos[:-1])
        ax_sub.set_xticklabels(['Q', 'E', 'V', 'F'], fontsize=9, fontweight='bold', color='black', alpha=1)
        ax_sub.set_theta_offset(np.pi / 2)
        ax_sub.set_theta_direction(-1)
        ax_sub.set_ylim(0, 10)
        ax_sub.set_yticks([])
        ax_sub.text(0.5, -0.3, f"  {nombre}  ", transform=ax_sub.transAxes, ha='center', va='center', fontsize=16, fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='none'))

    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.14, hspace=0.4)
    plt.savefig(f'Grafica_Comparativa_Arquitecturas_{file_suffix}.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica 5: Comparativa (Advanced Radar) guardada como 'Grafica_Comparativa_Arquitecturas_{file_suffix}.png'")

# --- GRÁFICA 6: COMPARATIVA DE COSTES LOGARÍTMICA ---
# Análisis desglosado de costes por arquitectura (Inputs, Outputs y Embeddings)
def grafica_costes():
    PRECIO_INPUT_1M, PRECIO_OUTPUT_1M, COST_EMBEDDING_OPENAI = 0.41706, 2.68168, 0.097
    COLOR_EMBEDDING_FACE, COLOR_EMBEDDING_EDGE, COLOR_TEXTO, COLOR_TOTAL = '#E0F2F1', '#00897B', '#333333', '#111111'

    def calcular_coste(ti, to): return (ti / 1_000_000) * PRECIO_INPUT_1M, (to / 1_000_000) * PRECIO_OUTPUT_1M

    ci1, co1 = calcular_coste(44575, 5845)
    ci2, co2 = calcular_coste(603739, 12067)
    ci3, co3 = calcular_coste(2828444, 4462)

    costes_input, costes_output, costes_emb = [ci1, ci2, ci3], [co1, co2, co3], [COST_EMBEDDING_OPENAI, 0, 0]
    costes_total = [ci1 + co1 + COST_EMBEDDING_OPENAI, ci2 + co2, ci3 + co3]

    mako_cmap = sns.color_palette('mako', as_cmap=True)
    colores = [mako_cmap(0.15), mako_cmap(0.5), mako_cmap(0.85)]

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#f8f8f8')
    ax.set_facecolor('#f2f2f2')
    ax.grid(axis='y', color='white', linewidth=1.5, zorder=0)
    ax.set_axisbelow(True)

    x, bw = np.arange(3), 0.25
    ax.bar(x - bw, costes_input, bw, color=colores, zorder=3, edgecolor='white', linewidth=0.8, label='Tokens Input (sólido)')
    ax.bar(x, costes_output, bw, color=colores, zorder=3, edgecolor='white', linewidth=0.8, hatch='///', alpha=0.9, label='Tokens Output (rayado)')
    if costes_emb[0] > 0:
        ax.bar(x[0] + bw, costes_emb[0], bw, color=COLOR_EMBEDDING_FACE, edgecolor=COLOR_EMBEDDING_EDGE, zorder=3, linewidth=1.2, hatch='..', alpha=1.0, label='Embedding OpenAI (Puntos)') 

    ax.set_yscale('log')
    from matplotlib.transforms import offset_copy

    trans_in = offset_copy(ax.transData, fig=fig, x=0, y=3, units='points')
    for xi, ci in zip(x, costes_input): ax.text(xi - bw, ci, f'{ci:.4f}€', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=COLOR_TEXTO, transform=trans_in)

    trans_out = offset_copy(ax.transData, fig=fig, x=0, y=3, units='points')
    for xi, co in zip(x, costes_output): ax.text(xi, co, f'{co:.4f}€', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=COLOR_TEXTO, transform=trans_out)

    trans_emb = offset_copy(ax.transData, fig=fig, x=0, y=4, units='points')
    if costes_emb[0] > 0:
        ax.text(x[0] + bw, COST_EMBEDDING_OPENAI, f'{costes_emb[0]:.4f}€', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=COLOR_TEXTO, transform=trans_emb)
        ax.annotate('Embedding OpenAI', xy=(x[0] + bw, costes_emb[0] * 0.8), xycoords='data', xytext=(-100, 0), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=COLOR_EMBEDDING_EDGE, lw=1.5), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_EMBEDDING_EDGE, alpha=0.95, lw=1.2), fontsize=9.5, fontweight='bold', color=COLOR_EMBEDDING_EDGE, ha='right', va='center')

    for i, (xi, total) in enumerate(zip(x, costes_total)):
        max_bar = max(costes_input[i], costes_output[i], costes_emb[i])
        trans = offset_copy(ax.transData, fig=fig, x=0, y=60, units='points')
        ax.text(xi, max_bar, f'Total: {total:.4f}€', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=COLOR_TOTAL, transform=trans, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaaaaa', alpha=0.9, lw=1))

    ref = costes_total[0] 
    for i, (xi, total) in enumerate(zip(x, costes_total)):
        if i > 0:
            mult = total / ref
            max_bar = max(costes_input[i], costes_output[i], costes_emb[i])
            trans_mult = offset_copy(ax.transData, fig=fig, x=0, y=100, units='points')
            ax.text(xi, max_bar, f'{mult:.1f}x más caro\nque RAG Pinecone', ha='center', va='bottom', fontsize=9.5, color='#c0392b', fontweight='bold', transform=trans_mult, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#c0392b', alpha=0.9, lw=1.2))

    ax.set_title('Comparativa de Coste Total por Arquitectura (20 Preguntas)', fontsize=17, fontweight='bold', pad=20)
    gran_total_rounded = np.ceil(sum(costes_total) * 100) / 100
    ax.text(0.5, 0.98, f'Coste Total: {gran_total_rounded:.2f}€', transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold', color='#2c3e50', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f6f3', edgecolor='#1abc9c', linewidth=1.5, alpha=0.95))
    ahorro_pct = (1 - ref / costes_total[2]) * 100
    ax.text(0.99, 0.98, f'RAG Pinecone reduce el coste un {ahorro_pct:.0f}%\nrespecto a Long Context', transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#555555', style='italic', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.85))

    ax.set_ylabel('Coste (€ EUR) — Escala Logarítmica', fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(['RAG Pinecone', 'RAG Native', 'Long Context'], fontsize=13, fontweight='bold')
    ax.set_ylim(bottom=None, top=max(costes_total) * 35) 
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colores[0], label='RAG Pinecone', edgecolor='black'), Patch(facecolor=colores[1], label='RAG Native', edgecolor='black'), Patch(facecolor=colores[2], label='Long Context', edgecolor='black'),
        Patch(facecolor='#dddddd', edgecolor='black', label='Input (Sólido)'), Patch(facecolor='#dddddd', edgecolor='black', hatch='///', label='Output (Rayado)'),
        Patch(facecolor=COLOR_EMBEDDING_FACE, edgecolor=COLOR_EMBEDDING_EDGE, hatch='..', label='Embedding (Puntos)', alpha=1.0),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.95, ncol=1, bbox_to_anchor=(0.01, 0.99)) 
    plt.tight_layout()
    plt.savefig('Comparativa_Costes_Final.png', dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()
    print("  Gráfica 6: Costes guardada como 'Comparativa_Costes_Final.png'")

# --- BLOQUE DE EXECUCIÓN PRINCIPAL ---
# Coordina la carga de datos y la generación secuencial de todas las visualizaciones
if __name__ == "__main__":
    print(" Generando gráficas de análisis de rendimiento...\n")
    
    # Cargar datos
    df1, df2, df3 = cargar_datos()
    
    # 1. Generar gráficas que no dependen del score (Latencia y Costes)
    print("\n--- Generando Gráficas Generales ---")
    grafica_latencia(df1, df2, df3)
    grafica_costes()
    
    # 2. Generar gráficas con EVALUACIÓN HUMANA (Tus notas manuales)
    print("\n--- Generando Gráficas (Evaluación Humana) ---")
    grafica_calidad(df1, df2, df3, is_judge=False)
    grafica_roi(df1, df2, df3, is_judge=False)
    grafica_calidad_contexto_combinada(df1, df2, df3, is_judge=False)
    grafica_comparativa_arquitecturas(df1, df2, df3, is_judge=False)
    
    # 3. Generar gráficas con EVALUACIÓN DEL JUEZ LLM (GPT-4o)
    print("\n--- Generando Gráficas (LLM-Judge) ---")
    grafica_calidad(df1, df2, df3, is_judge=True)
    grafica_roi(df1, df2, df3, is_judge=True)
    grafica_calidad_contexto_combinada(df1, df2, df3, is_judge=True)
    grafica_comparativa_arquitecturas(df1, df2, df3, is_judge=True)
    
    print("\n" + "="*60)
    print("  TODAS LAS GRÁFICAS GENERADAS EXITOSAMENTE")
    print("="*60)
    print("\nAhora tendrás dos versiones de las gráficas de puntuación:")
    print("  - Terminadas en '_Humana.png' (Tus notas)")
    print("  - Terminadas en '_Judge.png'  (Notas de GPT-4o)")