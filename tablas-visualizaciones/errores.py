# === Regenerar solo figuras de errores (figura3 y figura6) ===
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ruta del CSV general
OUT_DIR = Path(r"C:\Users\hered\Desktop\TFM\TFM\TFM2\tablas-visualizaciones")
CSV = OUT_DIR / "comparativa_completa_medcalc.csv"

# Leer datos
df = pd.read_csv(CSV)
plt.style.use('seaborn-v0_8')

# ==========================================================
# FIGURA 3 - Distribución de errores por tipo de estrategia
# ==========================================================
print("🎨 Generando figura 3: Distribución de errores por estrategia...")

strategy_error_data = df[['Estrategia', 'error_A_knowledge', 'error_B_extraction', 'error_C_arithmetic']].melt(
    id_vars='Estrategia',
    var_name='Tipo Error',
    value_name='Porcentaje (%)'
)

# Mapear nombres legibles
error_names = {
    'error_A_knowledge': 'A: Conocimiento',
    'error_B_extraction': 'B: Extracción',
    'error_C_arithmetic': 'C: Aritmético'
}
strategy_error_data['Tipo Error'] = strategy_error_data['Tipo Error'].map(error_names)

plt.figure(figsize=(12, 8))
sns.boxplot(
    data=strategy_error_data,
    x='Estrategia',
    y='Porcentaje (%)',
    hue='Tipo Error',
    palette='Set2'
)
plt.title('Distribución de Errores por Estrategia', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR / 'figura3_distribucion_errores.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ figura3_distribucion_errores.png generada correctamente")

# ==========================================================
# FIGURA 6 - Distribución promedio de tipos de error
# ==========================================================
print("🎨 Generando figura 6: Distribución promedio de tipos de error...")

error_means = df[['error_A_knowledge', 'error_B_extraction', 'error_C_arithmetic']].mean()

error_labels = [
    'A: No detecta\ninfo existente',
    'B: Extracción\nincorrecta',
    'C: Error\naritmético'
]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

plt.figure(figsize=(10, 6))
bars = plt.bar(error_labels, error_means.values, color=colors, alpha=0.7)
plt.ylabel('Porcentaje Promedio (%)')
plt.title('Distribución Promedio de Tipos de Error', fontsize=14, fontweight='bold')

for bar, value in zip(bars, error_means.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{value:.1f}%',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'figura6_distribucion_errores_promedio.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ figura6_distribucion_errores_promedio.png generada correctamente")

print("\n🎉 Regeneración de figuras de error completada.")
