# === Evaluación comparativa de modelos MEDCALC-Bench ===
# Analiza todos los modelos y estrategias de prompt

import os, numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ------------ Configuración ------------
OUT_DIR = "./outputs"
GT_CSV  = r"C:\Users\hered\Desktop\TFM\TFM\IMC2\valid_imc.csv"

# Tolerancias para evaluación
BMI_ABS_TOL = 0.1    # Tolerancia absoluta para BMI
BMI_PCT_TOL = 5      # Tolerancia 5% para BMI
H_TOL = 0.05         # Tolerancia altura (m)
W_TOL = 2.0          # Tolerancia peso (kg)

# ------------ Funciones de evaluación ------------
def _to_bool(series):
    """Convierte robustamente a bool"""
    return series.astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])

def load_gt():
    """Carga ground truth"""
    df = pd.read_csv(GT_CSV, dtype={"patient_id": str})
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    for c in ["height_m","weight_kg","BMI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def parse_filename(filename):
    """Extrae modelo y estrategia del nombre de archivo"""
    name = filename.stem
    parts = name.split('_')
    
    # Estrategia: v1_simple, v2_estricto, v3_fewshot, v4_encadenado
    strategy = f"{parts[1]}_{parts[2]}" if len(parts) > 2 else parts[1]
    
    # Modelo: desde la posición 3 hasta el final (excepto _n20)
    model_parts = parts[3:-1] if parts[-1].startswith('n') else parts[3:]
    model_name = '_'.join(model_parts)
    
    return strategy, model_name

def load_all_predictions():
    """Carga todas las predicciones disponibles"""
    pred_files = list(Path(OUT_DIR).glob("pred_*.csv"))
    all_predictions = {}
    
    for pred_file in pred_files:
        strategy, model_name = parse_filename(pred_file)
        
        df = pd.read_csv(pred_file, dtype={"patient_id": str})
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        
        # Normalizar nombres de columnas
        col_map = {
            'height_m_pred': 'height_m_pred',
            'weight_kg_pred': 'weight_kg_pred', 
            'BMI_pred_raw': 'bmi_pred',
            'BMI_from_pred_hw': 'bmi_from_hw_pred'
        }
        
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = pd.to_numeric(df[old], errors="coerce")
        
        key = f"{model_name}_{strategy}"
        all_predictions[key] = {
            'df': df,
            'model': model_name,
            'strategy': strategy,
            'file': pred_file.name
        }
    
    return all_predictions

def analyze_errors_per_sample(gt_row, pred_row):
    """Analiza errores para una muestra individual"""
    errors = {
        "error_A_knowledge": False,    # No detectó información existente
        "error_B_extraction": False,   # Extracción incorrecta
        "error_C_arithmetic": False,   # Error cálculo/conversión
        "h_ok": False,
        "w_ok": False, 
        "bmi_ok_abs": False,
        "bmi_ok_5p": False
    }
    
    # Verificar si hay datos en GT
    h_gt, w_gt, bmi_gt = gt_row["height_m"], gt_row["weight_kg"], gt_row["BMI"]
    h_pred, w_pred, bmi_pred = pred_row.get("height_m_pred"), pred_row.get("weight_kg_pred"), pred_row.get("bmi_pred")
    
    has_gt_data = pd.notna(h_gt) or pd.notna(w_gt) or pd.notna(bmi_gt)
    has_pred_data = pd.notna(h_pred) or pd.notna(w_pred) or pd.notna(bmi_pred)
    
    # Error A: No detectó información existente
    if has_gt_data and not has_pred_data:
        errors["error_A_knowledge"] = True
        return errors, "A_knowledge"
    
    if not has_gt_data and not has_pred_data:
        return errors, "ok"
    
    # Verificar exactitud por campo
    if pd.notna(h_gt) and pd.notna(h_pred):
        errors["h_ok"] = abs(h_gt - h_pred) <= H_TOL
    elif pd.notna(h_gt) and pd.isna(h_pred):
        errors["error_A_knowledge"] = True
    
    if pd.notna(w_gt) and pd.notna(w_pred):
        errors["w_ok"] = abs(w_gt - w_pred) <= W_TOL  
    elif pd.notna(w_gt) and pd.isna(w_pred):
        errors["error_A_knowledge"] = True
    
    if pd.notna(bmi_gt) and pd.notna(bmi_pred):
        errors["bmi_ok_abs"] = abs(bmi_gt - bmi_pred) <= BMI_ABS_TOL
        errors["bmi_ok_5p"] = abs(bmi_gt - bmi_pred) / bmi_gt * 100 <= BMI_PCT_TOL
    elif pd.notna(bmi_gt) and pd.isna(bmi_pred):
        errors["error_A_knowledge"] = True
    
    # Error B: Extracción incorrecta (valores fuera de tolerancia)
    if (pd.notna(h_gt) and pd.notna(h_pred) and not errors["h_ok"]) or \
       (pd.notna(w_gt) and pd.notna(w_pred) and not errors["w_ok"]):
        errors["error_B_extraction"] = True
    
    # Error C: Error aritmético (BMI incorrecto cuando H y W son correctos)
    bmi_from_hw = None
    if pd.notna(h_pred) and pd.notna(w_pred) and h_pred > 0:
        bmi_from_hw = w_pred / (h_pred * h_pred)
    
    if bmi_from_hw and pd.notna(bmi_gt) and pd.notna(bmi_pred):
        if errors["h_ok"] and errors["w_ok"] and abs(bmi_gt - bmi_pred) > BMI_ABS_TOL:
            errors["error_C_arithmetic"] = True
    
    # Determinar etiqueta primaria
    primary_error = determine_primary_error(errors, has_gt_data)
    
    return errors, primary_error

def determine_primary_error(errors, has_gt_data):
    """Determina la etiqueta de error primaria según prioridad"""
    if not has_gt_data:
        return "ok"
    
    if errors["error_A_knowledge"]:
        return "A_knowledge"
    elif errors["error_B_extraction"]:
        return "B_extraction" 
    elif errors["error_C_arithmetic"]:
        return "C_arithmetic"
    elif all([errors["h_ok"] if pd.notna(errors["h_ok"]) else True,
              errors["w_ok"] if pd.notna(errors["w_ok"]) else True,
              errors["bmi_ok_abs"] if pd.notna(errors["bmi_ok_abs"]) else True]):
        return "ok"
    else:
        return "D_other"

def generate_medcalc_report(gt_df, pred_df, model_name, strategy):
    """Genera reporte MedCalc-Bench para un modelo y estrategia"""
    # Merge con GT
    merged = pred_df.merge(gt_df[["patient_id", "height_m", "weight_kg", "BMI"]], 
                          on="patient_id", how="inner")
    
    report_rows = []
    
    for _, row in merged.iterrows():
        errors, primary_error = analyze_errors_per_sample(row, row)
        
        report_row = {
            "patient_id": row["patient_id"],
            "model": model_name,
            "strategy": strategy,
            "height_gt": row["height_m"],
            "weight_gt": row["weight_kg"], 
            "bmi_gt": row["BMI"],
            "height_pred": row.get("height_m_pred"),
            "weight_pred": row.get("weight_kg_pred"),
            "bmi_pred": row.get("bmi_pred"),
            "primary_error": primary_error,
            "bmi_explicit_in_note": pd.notna(row["BMI"])  # Asumimos que si hay BMI en GT, está explícito
        }
        
        # Añadir errores y métricas de correctitud
        report_row.update(errors)
        
        # Añadir métricas compuestas
        report_row["predijo_bien_all"] = all([
            report_row["h_ok"] if pd.notna(row["height_m"]) else True,
            report_row["w_ok"] if pd.notna(row["weight_kg"]) else True, 
            report_row["bmi_ok_abs"] if pd.notna(row["BMI"]) else True
        ])
        
        report_row["predijo_bien_bmi"] = report_row["bmi_ok_abs"]
        
        report_rows.append(report_row)
    
    return pd.DataFrame(report_rows)

def calculate_model_metrics(report_df):
    """Calcula métricas agregadas para un modelo"""
    if len(report_df) == 0:
        return {}
    
    N = len(report_df)
    
    def pct(col):
        if col not in report_df.columns:
            return np.nan
        return 100.0 * report_df[col].mean()
    
    metrics = {
        "N": N,
        "bmi_explicit_in_note": pct("bmi_explicit_in_note"),
        "error_A_knowledge": pct("error_A_knowledge"),
        "error_B_extraction": pct("error_B_extraction"), 
        "error_C_arithmetic": pct("error_C_arithmetic"),
        "h_ok": pct("h_ok"),
        "w_ok": pct("w_ok"),
        "bmi_ok_abs": pct("bmi_ok_abs"),
        "bmi_ok_5p": pct("bmi_ok_5p"),
        "predijo_bien_all": pct("predijo_bien_all"),
        "predijo_bien_bmi": pct("predijo_bien_bmi")
    }
    
    # Distribución de errores primarios
    if "primary_error" in report_df.columns:
        error_dist = report_df["primary_error"].value_counts(normalize=True) * 100
        primary_order = ["B_extraction", "A_knowledge", "ok", "D_other", "C_arithmetic", "incomplete"]
        for error_type in primary_order:
            metrics[f"primary_{error_type}"] = error_dist.get(error_type, 0.0)
    
    return metrics

def create_comparison_table(all_metrics):
    """Crea tabla comparativa de todos los modelos y estrategias"""
    # Métricas a incluir en la tabla
    metric_columns = [
        "N",
        "bmi_explicit_in_note",
        "error_A_knowledge", "error_B_extraction", "error_C_arithmetic",
        "h_ok", "w_ok", "bmi_ok_abs", "bmi_ok_5p", 
        "predijo_bien_all", "predijo_bien_bmi",
        "primary_ok", "primary_A_knowledge", "primary_B_extraction", 
        "primary_C_arithmetic", "primary_D_other"
    ]
    
    rows = []
    for config_name, metrics in all_metrics.items():
        model, strategy = config_name.split('_', 1) if '_' in config_name else (config_name, 'unknown')
        row = {
            "Modelo": model,
            "Estrategia": strategy,
            "Configuración": config_name
        }
        for col in metric_columns:
            row[col] = metrics.get(col, np.nan)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ordenar por accuracy general (predijo_bien_all)
    if "predijo_bien_all" in df.columns:
        df = df.sort_values("predijo_bien_all", ascending=False)
    
    return df

# ------------ Ejecución principal ------------
def main():
    print("🔍 Evaluando todos los modelos MEDCALC-Bench...")
    
    # Cargar Ground Truth
    gt_df = load_gt()
    print(f"✅ GT cargado: {len(gt_df)} registros")
    
    # Cargar todas las predicciones
    all_predictions = load_all_predictions()
    print(f"📁 Archivos de predicción encontrados: {len(all_predictions)}")
    
    all_reports = {}
    all_metrics = {}
    
    for config_name, pred_info in all_predictions.items():
        model = pred_info['model']
        strategy = pred_info['strategy']
        
        print(f"\n📊 Procesando: {model} - {strategy}")
        print(f"   Archivo: {pred_info['file']}")
        
        # Generar reporte MedCalc
        report_df = generate_medcalc_report(gt_df, pred_info['df'], model, strategy)
        all_reports[config_name] = report_df
        
        # Calcular métricas
        metrics = calculate_model_metrics(report_df)
        all_metrics[config_name] = metrics
        
        # Guardar reporte individual
        report_path = Path(OUT_DIR) / f"report_medcalc_{config_name}.csv"
        report_df.to_csv(report_path, index=False)
        print(f"   ✅ Reporte guardado: {report_path}")
        
        # Mostrar resumen rápido
        if metrics:
            print(f"   📈 Accuracy All: {metrics.get('predijo_bien_all', 0):.1f}%")
            print(f"   📈 Accuracy BMI: {metrics.get('bmi_ok_abs', 0):.1f}%")
            print(f"   🔍 Errores: A={metrics.get('error_A_knowledge', 0):.1f}%, "
                  f"B={metrics.get('error_B_extraction', 0):.1f}%, "
                  f"C={metrics.get('error_C_arithmetic', 0):.1f}%")
    
    # Generar tabla comparativa
    if all_metrics:
        comparison_table = create_comparison_table(all_metrics)
        
        # Guardar tabla comparativa
        comp_path = Path(OUT_DIR) / "comparativa_completa_medcalc.csv"
        comparison_table.to_csv(comp_path, index=False)
        
        print(f"\n🎯 TABLA COMPARATIVA COMPLETA GUARDADA: {comp_path}")
        print("\n" + "="*100)
        print("COMPARATIVA COMPLETA - TODOS LOS MODELOS Y ESTRATEGIAS")
        print("="*100)
        
        # Mostrar tabla resumida formateada
        display_cols = ["Modelo", "Estrategia", "predijo_bien_all", "predijo_bien_bmi", 
                       "error_A_knowledge", "error_B_extraction", "error_C_arithmetic",
                       "primary_ok"]
        
        display_df = comparison_table[display_cols].copy()
        
        # Formatear porcentajes
        for col in display_cols[2:]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        print(display_df.to_string(index=False))
        
        # Análisis por modelo (mejor estrategia por modelo)
        print(f"\n🏆 MEJORES ESTRATEGIAS POR MODELO:")
        best_by_model = comparison_table.loc[comparison_table.groupby("Modelo")["predijo_bien_all"].idxmax()]
        for _, row in best_by_model.iterrows():
            print(f"   • {row['Modelo']:25} - {row['Estrategia']:15}: {row['predijo_bien_all']:.1f}% accuracy")
        
        # Análisis por estrategia (mejor modelo por estrategia)
        print(f"\n🏆 MEJORES MODELOS POR ESTRATEGIA:")
        best_by_strategy = comparison_table.loc[comparison_table.groupby("Estrategia")["predijo_bien_all"].idxmax()]
        for _, row in best_by_strategy.iterrows():
            print(f"   • {row['Estrategia']:15} - {row['Modelo']:25}: {row['predijo_bien_all']:.1f}% accuracy")
        
        # Top 5 configuraciones generales
        print(f"\n🏆 TOP 5 CONFIGURACIONES GENERALES:")
        top5 = comparison_table.nlargest(5, "predijo_bien_all")[["Modelo", "Estrategia", "predijo_bien_all"]]
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"   {i}. {row['Modelo']:25} - {row['Estrategia']:15}: {row['predijo_bien_all']:.1f}%")
    
    else:
        print("❌ No se pudieron generar métricas para ningún modelo")

if __name__ == "__main__":
    main()