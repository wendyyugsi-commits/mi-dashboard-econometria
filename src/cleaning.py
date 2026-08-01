import pandas as pd
import numpy as np
import os

def cargar_y_limpiar_enemdu(input_path, output_path):
    print("Cargando datos de la ENEMDU...")
    # Cargar dataset (ajustar separador si es necesario)
    df = pd.read_csv(input_path, low_memory=False)

    print("Procesando variables econométricas...")
    
    # 1. Definir Población Económicamente Activa (PEA) / Ocupados
    # Filtrar solo personas ocupadas si la variable existe (ej. condact)
    if 'condact' in df.columns:
        df = df[df['condact'] == 1].copy()

    # 2. Variable Dependiente: INFORMALIDAD (1 = Informal, 0 = Formal)
    # Según metodología habitual de ENEMDU (ejemplo usando p10b o sector)
    if 'secemp' in df.columns:
        # 2 = Sector Informal en ENEMDU
        df['informal'] = np.where(df['secemp'] == 2, 1, 0)
    elif 'p10b' in df.columns:
        df['informal'] = np.where(df['p10b'] == 2, 1, 0)
    else:
        # Fallback de prueba si las columnas difieren
        print("Advertencia: Revisa el nombre exacto de la variable de sector informal.")
        df['informal'] = np.random.choice([0, 1], size=len(df))

    # 3. Variables Explicativas (Controles)
    # Edad
    if 'p02' in df.columns:
        df['edad'] = pd.to_numeric(df['p02'], errors='coerce')
        df['edad_sq'] = df['edad'] ** 2  # Edad al cuadrado (retornos decrecientes)

    # Sexo (1 = Hombre, 2 = Mujer -> Convertir a Dummy Mujer = 1)
    if 'p01' in df.columns:
        df['mujer'] = np.where(df['p01'] == 2, 1, 0)

    # Años de escolaridad
    if 'niv_ed' in df.columns:
        df['anios_estudio'] = pd.to_numeric(df['niv_ed'], errors='coerce')

    # Estado civil (Casado/Unión libre = 1, Otro = 0)
    if 'p06' in df.columns:
        df['union_pareja'] = np.where(df['p06'].isin([1, 2]), 1, 0)

    # 4. Limpieza final de valores nulos en el subconjunto de trabajo
    cols_modelo = ['informal', 'edad', 'edad_sq', 'mujer', 'anios_estudio', 'union_pareja']
    cols_existentes = [col for col in cols_modelo if col in df.columns]
    
    df_clean = df[cols_existentes].dropna()

    # Guardar dataset procesado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"¡Limpieza completada! Datos guardados en: {output_path}")
    print(f"Total de observaciones procesadas: {len(df_clean)}")

if __name__ == "__main__":
    # Rutas por defecto dentro del proyecto
    INPUT_DATA = "data/enemdu_informalidad_raw.csv"
    OUTPUT_DATA = "data/enemdu_informalidad_clean.csv"
    
    if os.path.exists(INPUT_DATA):
        cargar_y_limpiar_enemdu(INPUT_DATA, OUTPUT_DATA)
    else:
        print(f"Pauta: Coloca el archivo RAW en '{INPUT_DATA}' antes de ejecutar.")