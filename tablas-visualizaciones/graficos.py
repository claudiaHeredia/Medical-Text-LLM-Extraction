import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# === 1. Directorio de entrada ===
CARPETA = Path(r"C:\Users\hered\Desktop\TFM\TFM\TFM2\tablas-visualizaciones\graficos_modelos")
SALIDA = CARPETA / "graficos_modelos"
SALIDA.mkdir(exist_ok=True)

# === 2. Columnas a representar (ahora incluye Error C) ===
cols_to_plot = [
    "predijo_bien_all",      # Acierto total
    "predijo_bien_bmi",      # Acierto IMC
    "h_ok",                  # Altura correcta
    "w_ok",                  # Peso correcto
    "error_A_knowledge",     # Error tipo A
    "error_B_extraction",    # Error tipo B
    "error_C_arithmetic"     # Error tipo C
]

labels = [
    "Acierto total",
    "Acierto IMC",
    "Altura correcta",
    "Peso correcto",
    "Error A (conocimiento)",
    "Error B (extracción)",
    "Error C (aritmético)"
]

# Paleta de colores diferenciada
colors = [
    "#f4a261",  # naranja suave
    "#2a9d8f",  # verde azulado
    "#90be6d",  # verde claro
    "#3a86ff",  # azul brillante
    "#ffca3a",  # amarillo
    "#577590",  # azul grisáceo
    "#ef476f"   # rojo rosado (nuevo para error C)
]

# === 3. Buscar archivos ===
archivos = sorted([f for f in CARPETA.glob("tabla_individual_*.csv")])

print(f"Archivos encontrados: {len(archivos)}")
if not archivos:
    print("⚠️ No se encontraron archivos. Verifica la ruta.")
else:
    for f in archivos:
        print("📄", f.name)

# === 4. Generar gráfico por modelo ===
for archivo in archivos:
    try:
        df = pd.read_csv(archivo)
        modelo = df["Modelo"].iloc[0] if "Modelo" in df.columns else archivo.stem.replace("tabla_individual_", "")

        x = np.arange(len(df))
        width = 0.11  # más estrecho para que quepan todas las barras

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, col in enumerate(cols_to_plot):
            if col in df.columns:
                ax.bar(x + i * width - width * 3, df[col], width, label=labels[i], color=colors[i])
            else:
                print(f"⚠️ {col} no encontrado en {archivo.name}")

        # === Estética ===
        ax.set_title(f"Desempeño del modelo {modelo} por tipo de prompt", fontsize=13, weight="bold")
        ax.set_xlabel("Estrategia", fontsize=11)
        ax.set_ylabel("Porcentaje (%)", fontsize=11)
        ax.set_xticks(x)

        if "Estrategia" in df.columns:
            etiquetas = [e.replace("v1_", "").replace("v2_", "").replace("v3_", "").replace("v4_", "") for e in df["Estrategia"]]
            ax.set_xticklabels(etiquetas, fontsize=10)
        else:
            ax.set_xticklabels([f"v{i+1}" for i in range(len(df))], fontsize=10)

        ax.legend(title=None, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()

        # === 5. Guardar gráfico ===
        salida_img = SALIDA / f"grafico_{modelo.replace('/', '_')}.png"
        plt.savefig(salida_img, dpi=300)
        plt.close()
        print(f"✅ Gráfico guardado: {salida_img}")

    except Exception as e:
        print(f"❌ Error procesando {archivo.name}: {e}")
