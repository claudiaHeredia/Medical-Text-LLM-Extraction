import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# ==============================
# 1️⃣ Datos del proyecto (por meses)
# ==============================
tareas = [
    ["Investigación y revisión bibliográfica", 2, 4, 2, 7],
    ["Preparación y validación de datos", 6, 6, 6, 7],
    ["Configuración y pruebas de entorno local", 6, 7, 6, 9],
    ["Desarrollo experimental y ejecución de modelos", 7, 8, 7, 9],
    ["Evaluación y clasificación de errores", 8, 9, 8, 10],
    ["Optimización y verificación de prompts", 8, 10, 8, 10],
    ["Visualización y análisis de resultados", 9, 10, 9, 10],
    ["Documentación y redacción final", 6, 9, 6, 10],
    ["Reuniones y revisiones con tutor", 2, 10, 2, 10]
]

df = pd.DataFrame(tareas, columns=[
    "Tarea", "MesInicio_Plan", "MesFin_Plan", "MesInicio_Real", "MesFin_Real"
])

# ==============================
# 2️⃣ Configuración de estilo
# ==============================
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11.5, 6.5))

color_plan = "#4C72B0"   # azul profesional
color_real = "#DD8452"   # naranja sobrio
color_pause = "#B0B0B0"  # gris para la pausa

# ==============================
# 3️⃣ Dibujar las barras
# ==============================
y_pos = range(len(df))

for i, row in df.iterrows():
    # Planificación (azul)
    ax.barh(i, row["MesFin_Plan"] - row["MesInicio_Plan"] + 1,
            left=row["MesInicio_Plan"], height=0.35,
            color=color_plan, edgecolor='none', alpha=0.8)
    # Ejecución real (naranja)
    ax.barh(i, row["MesFin_Real"] - row["MesInicio_Real"] + 1,
            left=row["MesInicio_Real"], height=0.20,
            color=color_real, edgecolor='none', alpha=0.9)

# ==============================
# 4️⃣ Pausa (abril–mayo)
# ==============================
ax.axvspan(3, 6, color=color_pause, alpha=0.15)
ax.text(4.5, -0.4, "Pausa / Replanificación", fontsize=8.5,
        color='gray', ha='center', va='center', style='italic')

# ==============================
# 5️⃣ Estilo y formato general
# ==============================
ax.set_yticks(y_pos)
ax.set_yticklabels(df["Tarea"], fontsize=9.5)
ax.invert_yaxis()

# Ejes y etiquetas
ax.set_xlabel("Meses del año 2025", fontsize=10, labelpad=10)
ax.set_title("Diagrama de Gantt — Planificación inicial vs Ejecución real",
             fontsize=13, fontweight='bold', pad=14)

# Eje X con nombres de meses
meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
ax.set_xticks(range(1, 13))
ax.set_xticklabels(meses, fontsize=9)
ax.set_xlim(1, 12)

# Línea vertical de entrega (noviembre)
ax.axvline(11, color='red', linestyle='--', lw=1.3)
ax.text(11, len(df)+0.3, "Entrega del TFM (3 nov)",
        color='red', fontsize=9.5, va='center', ha='center', fontweight='semibold')

# Leyenda elegante
plan_patch = mpatches.Patch(color=color_plan, label='Planificación inicial')
real_patch = mpatches.Patch(color=color_real, label='Ejecución real')
plt.legend(handles=[plan_patch, real_patch],
           loc='upper right', fontsize=9.5, frameon=True, edgecolor='gray')

# Cuadrícula sutil
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.grid(axis='y', linestyle=':', alpha=0.15)

# Eliminar bordes extra
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("gantt_planificacion_elegante.png", dpi=300, bbox_inches='tight')
plt.show()
