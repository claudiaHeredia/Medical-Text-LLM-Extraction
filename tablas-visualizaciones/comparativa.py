# ==========================================================
# 🧩 REGENERAR CSVs GLOBALES PRESERVANDO TODAS LAS COLUMNAS
# ==========================================================
import pandas as pd
from pathlib import Path

CARPETA = Path(r"C:\Users\hered\Desktop\TFM\TFM\TFM2\tablas-visualizaciones")

# Buscar solo las tablas individuales
archivos = sorted(CARPETA.glob("tabla_individual_*.csv"))
print(f"📄 Tablas individuales encontradas: {len(archivos)}")

dfs = []
for f in archivos:
    df = pd.read_csv(f)

    # Normalizar encabezados (evita espacios invisibles)
    df.columns = df.columns.str.strip()

    # Normalizar nombres de estrategia (para evitar "v1_simple " o mayúsculas)
    if "Estrategia" in df.columns:
        df["Estrategia"] = df["Estrategia"].astype(str).str.strip()

    # Si falta columna 'N', no forzar a 20 (deja NaN o el valor real)
    if "N" not in df.columns:
        df["N"] = pd.NA

    # Si falta columna 'Modelo', intentar inferir desde el nombre del archivo
    if "Modelo" not in df.columns:
        nombre_modelo = f.stem.replace("tabla_individual_", "").replace("_", "-")
        df["Modelo"] = nombre_modelo

    dfs.append(df)

# Concatenar todas las tablas individuales
df_full = pd.concat(dfs, ignore_index=True, sort=False)

# Asegurar orden lógico de columnas (Modelo, Estrategia primero)
cols_ordenadas = sorted(df_full.columns)
if "Modelo" in cols_ordenadas:
    cols_ordenadas.remove("Modelo")
if "Estrategia" in cols_ordenadas:
    cols_ordenadas.remove("Estrategia")
df_full = df_full[["Modelo", "Estrategia"] + cols_ordenadas]

# Guardar CSV principal completo
csv_completa = CARPETA / "comparativa_completa_medcalc.csv"
df_full.to_csv(csv_completa, index=False)
print(f"✅ CSV general guardado correctamente: {csv_completa}")

# --- CSV agregado por modelo (promedio numérico) ---
df_modelos = (
    df_full.groupby("Modelo")
    .mean(numeric_only=True)
    .reset_index()
)
csv_modelos = CARPETA / "comparativa_modelos_medcalc.csv"
df_modelos.to_csv(csv_modelos, index=False)
print(f"✅ CSV por modelo guardado: {csv_modelos}")

# --- Diagnóstico rápido ---
print("\n🔎 Diagnóstico post-concatenación:")
print(f"Filas totales: {len(df_full)}")
print("Modelos:", ", ".join(sorted(df_full['Modelo'].unique())))
print("Estrategias:", ", ".join(sorted(df_full['Estrategia'].unique())))
