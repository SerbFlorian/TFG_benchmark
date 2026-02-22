import pandas as pd
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno desde .env.local
load_dotenv(".env.local")

# 1. CONFIGURACIÓN
API_KEY = os.getenv("OPENAI_API_KEY")
EXCEL_FILE = "TFG_Benchmark_Questions.xlsx"

# Inicializamos el cliente de OpenAI
client = OpenAI(api_key=API_KEY)

# Arquitecturas a evaluar
arquitecturas = ['Arch1', 'Arch2', 'Arch3']

def evaluar_respuesta(pregunta, ground_truth, respuesta_generada):
    """Llama a GPT-4o para que actúe como juez estricto."""
    
    # Si no hay respuesta generada, devolvemos un 0 directo
    if pd.isna(respuesta_generada) or str(respuesta_generada).strip() == "":
        return 0.0, "No hay respuesta generada."

    prompt_sistema = """Eres un ingeniero mecánico experto y un evaluador estricto. Tu tarea es evaluar la precisión (Accuracy) de la respuesta generada por un sistema de IA comparándola con la respuesta real (Ground Truth) de un manual de taller de BMW.

Evalúa del 1 al 10 basándote en esta rúbrica:
- 10: La respuesta es perfecta, contiene todos los datos técnicos (pares de apriete, medidas) exactos y no tiene información extraña.
- 7-9: La respuesta es correcta y útil, pero omite algún detalle menor o es redundante.
- 4-6: La respuesta tiene información parcialmente correcta, pero omite el dato clave o es ambigua.
- 1-3: La respuesta es incorrecta, alucina datos técnicos peligrosos o dice que no encuentra la información.

Devuelve ÚNICAMENTE un formato JSON válido con dos claves: 
"razonamiento" (string, una frase breve justificando la nota) y 
"puntuacion" (número entero o decimal del 1 al 10)."""

    prompt_usuario = f"""
**Pregunta:** {pregunta}
**Respuesta Real (Ground Truth):** {ground_truth}
**Respuesta Generada:** {respuesta_generada}
"""

    try:
        # Llamada a la API forzando formato JSON
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.0 # Temperatura 0 para que sea lo más objetivo posible
        )
        
        # Extraer y parsear el JSON
        resultado_str = response.choices[0].message.content
        resultado_json = json.loads(resultado_str)
        
        puntuacion = float(resultado_json.get("puntuacion", 0))
        razonamiento = str(resultado_json.get("razonamiento", ""))
        
        return puntuacion, razonamiento
        
    except Exception as e:
        print(f"Error en la API: {e}")
        return 0.0, f"Error en evaluación: {str(e)}"

# 2. BUCLE PRINCIPAL
hojas_procesadas = {}

for arch in arquitecturas:
    print(f"\n--- Iniciando evaluación para {arch} ---")
    
    # Leer la hoja del Excel
    df = pd.read_excel(EXCEL_FILE, sheet_name=arch)
    
    # Asegurarnos de usar los nombres de columnas exactos que pediste
    col_score = f"{arch}_Question_score_Judge(0/10)"
    col_reason = f"{arch}_Judge_Reasoning"
    
    if col_score not in df.columns:
        df[col_score] = 0.0
    if col_reason not in df.columns:
        df[col_reason] = df[col_reason].astype(object)
        
    # Columna específica donde está la respuesta de la IA
    col_response = f"{arch}_Response"
    
    # Iterar solo por las 20 primeras preguntas
    for index, row in df.head(20).iterrows():
        id_pregunta = row['ID']
        pregunta = row['Question']
        ground_truth = row['Reference_Answer']
        respuesta = row[col_response]
        
        print(f"[{arch}] Evaluando P{id_pregunta}...")
        
        # Llamar al juez
        puntuacion, razonamiento = evaluar_respuesta(pregunta, ground_truth, respuesta)
        
        # Guardar en el DataFrame
        df.at[index, col_score] = puntuacion
        df.at[index, col_reason] = razonamiento
        
        # Pausa para no saturar la API
        time.sleep(1) 
        
    hojas_procesadas[arch] = df
    print(f" Hoja {arch} completada.")

# 3. GUARDAR LOS CAMBIOS DIRECTAMENTE EN EL MISMO EXCEL
print(f"\nGuardando resultados directamente en '{EXCEL_FILE}'...")

# IMPORTANTE: Usamos mode='a' e if_sheet_exists='replace' para no borrar las otras hojas de tu Excel
with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    for sheet_name, df_processed in hojas_procesadas.items():
        df_processed.to_excel(writer, sheet_name=sheet_name, index=False)

print(f" ¡Proceso terminado! Abre el archivo '{EXCEL_FILE}' para ver los resultados.")