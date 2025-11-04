# improvement_strategy.py
import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_best_performers(results_df):
    """Analizar los mejores resultados para enfocar las mejoras"""
    
    # Filtrar solo modelos con al menos 10% de precisión
    viable_models = results_df[results_df['predijo_bien_bmi_pct'] >= 10.0]
    
    best_models = viable_models.nlargest(5, 'predijo_bien_bmi_pct')
    
    print(" MEJORES MODELOS IDENTIFICADOS:")
    print("=" * 60)
    
    for _, row in best_models.iterrows():
        print(f" {row['Modelo']} - {row['Estrategia']}")
        print(f"   Precisión BMI: {row['predijo_bien_bmi_pct']}%")
        print(f"   Precisión General: {row['predijo_bien_all_pct']}%")
        print(f"   Errores: A({row['error_A_knowledge_pct']}%) "
              f"B({row['error_B_extraction_pct']}%) "
              f"C({row['error_C_arithmetic_pct']}%)")
        print()
    
    return best_models

def create_targeted_improvements(best_models):
    """Crear mejoras específicas para los mejores modelos"""
    
    improvements = {}
    
    for _, model in best_models.iterrows():
        model_name = model['Modelo']
        strategy = model['Estrategia']
        errors = {
            'A': model['error_A_knowledge_pct'],
            'B': model['error_B_extraction_pct'], 
            'C': model['error_C_arithmetic_pct']
        }
        
        improvements[model_name] = {
            'current_performance': {
                'bmi_accuracy': model['predijo_bien_bmi_pct'],
                'overall_accuracy': model['predijo_bien_all_pct'],
                'main_errors': errors
            },
            'recommended_improvements': []
        }
        
        # Mejoras específicas basadas en tipo de error
        if errors['B'] > 0:  # Error de extracción
            improvements[model_name]['recommended_improvements'].append({
                'type': 'extraction_error',
                'description': 'Mejorar detección de unidades y valores',
                'actions': [
                    'Agregar ejemplos de extracción en few-shot',
                    'Incluir patrones regex como respaldo',
                    'Aumentar ventanas de contexto',
                    'Post-procesamiento con validación de rangos'
                ]
            })
        
        if errors['C'] > 0:  # Error aritmético
            improvements[model_name]['recommended_improvements'].append({
                'type': 'arithmetic_error', 
                'description': 'Mejorar cálculos de BMI',
                'actions': [
                    'Forzar cálculo automático de BMI',
                    'Validar consistencia altura-peso-BMI',
                    'Usar fórmula exacta en prompt',
                    'Post-calculation verification'
                ]
            })
        
        if errors['A'] > 0:  # Error de conocimiento
            improvements[model_name]['recommended_improvements'].append({
                'type': 'knowledge_error',
                'description': 'Mejorar comprensión de contexto médico',
                'actions': [
                    'Agregar ejemplos médicos específicos',
                    'Incluir explicaciones de conversiones',
                    'Contexto bilingüe español-inglés'
                ]
            })
    
    return improvements

def generate_enhanced_prompts():
    """Generar prompts mejorados basados en el análisis"""
    
    enhanced_prompts = {
        # MEJORA PARA v2_estricto (el mejor actual)
        "v2_estricto_mejorado": {
            "system": """Eres un experto en extracción de datos clínicos con validación automática.

EXTRACCIÓN PRIMARIA:
1. BUSCAR: altura (cm/m/ft), peso (kg/lb), BMI/IMC
2. CONVERTIR: 
   - Altura: cm→/100, ft→*0.3048, in→*0.0254  
   - Peso: lb→/2.205, stone→*6.35
3. VALIDAR RANGOS:
   - Altura: 1.2-2.2 m
   - Peso: 30-300 kg  
   - BMI: 10-80

VALIDACIÓN AUTOMÁTICA:
• Si altura y peso existen → CALCULAR BMI = peso / (altura²)
• Comparar con BMI del texto
• Priorizar cálculo si diferencia > 0.5

RESPUESTA SOLO JSON: {"height_m": número, "weight_kg": número, "bmi": número}""",
            
            "description": "v2_estricto mejorado con validación automática y cálculos"
        },
        
        # PROMEPT ESPECÍFICO PARA ERRORES DE EXTRACCIÓN
        "v7_extraction_focused": {
            "system": """ESPECIALISTA EN DETECCIÓN DE UNIDADES MÉDICAS

ENFOQUE EN PATRONES:
 ALTURA: 
   - "165 cm", "1.65 m", "5'6\\"", "5 pies 6 pulgadas"
   - "estatura", "talla", "mide", "altura"

 PESO:
   - "70 kg", "154 lb", "11 stone"  
   - "pesa", "peso", "kilogramos"

 BMI:
   - "BMI 25.7", "IMC 24.5", "índice de masa corporal"

CONVERSIONES EXPLÍCITAS:
• 5'4" = 1.63 m | 6'2" = 1.88 m
• 150 lb = 68.0 kg | 70 kg = 70.0 kg

RESPUESTA EXACTA EN JSON, sin texto adicional.""",
            
            "description": "Enfoque ultra-específico en detección de unidades"
        },
        
        # PROMEPT CON POST-CÁLCULO
        "v8_auto_calculate": {
            "system": """EXTRACTOR + CALCULADOR AUTOMÁTICO

PASO 1: Extraer valores crudos
PASO 2: Aplicar conversiones de unidades  
PASO 3: CALCULAR BMI automáticamente si hay altura y peso
PASO 4: Validar consistencia

FÓRMULA BMI: BMI = peso_kg / (altura_m * altura_m)

EJEMPLOS:
• "170 cm, 65 kg" → altura: 1.70, peso: 65.0 → BMI: 22.49
• "5'9\\", 160 lb" → altura: 1.75, peso: 72.6 → BMI: 23.73

SIEMPRE calcular BMI cuando sea posible.
RESPUESTA: {"height_m": valor, "weight_kg": valor, "bmi": valor}""",
            
            "description": "Forzar cálculo automático de BMI"
        }
    }
    
    return enhanced_prompts

def create_implementation_plan(best_models, improvements, enhanced_prompts):
    """Crear plan de implementación concreto"""
    
    print("\n PLAN DE IMPLEMENTACIÓN:")
    print("=" * 60)
    
    plan = {
        "phase_1": {
            "name": "Optimización Rápida - Phi-3-mini",
            "target": "Phi-3-mini-4k-instruct",
            "actions": [
                "Probar v2_estricto_mejorado (validación automática)",
                "Probar v7_extraction_focused (enfoque unidades)", 
                "Probar v8_auto_calculate (cálculo forzado)",
                "Comparar con v2_estricto original"
            ],
            "expected_improvement": "Reducir 5% error extracción → 97-98% precisión"
        },
        
        "phase_2": {
            "name": "Mejora StableLM-Zephyr", 
            "target": "stablelm-zephyr-3b",
            "actions": [
                "Aplicar mejores prompts de Phase 1",
                "Enfoque en reducir 45% error extracción",
                "Validación de rangos más estricta"
            ],
            "expected_improvement": "Mejorar de 40% a 60-70% precisión"
        },
        
        "phase_3": {
            "name": "Rescate Qwen2.5",
            "target": "Qwen2.5-0.5B-Instruct", 
            "actions": [
                "Prompts ultra-estructurados",
                "Few-shot con ejemplos muy específicos",
                "Post-procesamiento agresivo"
            ],
            "expected_improvement": "De 10% a 30-40% precisión"
        }
    }
    
    return plan

def main():
    # Cargar tus resultados
    results_df = pd.read_csv("./resultados_completos.csv")  # Ajusta la ruta
    
    print(" ESTRATEGIA DE MEJORA BASADA EN RESULTADOS")
    print("=" * 70)
    
    # 1. Identificar mejores performers
    best_models = analyze_best_performers(results_df)
    
    # 2. Análisis de mejoras específicas
    improvements = create_targeted_improvements(best_models)
    
    # 3. Generar prompts mejorados
    enhanced_prompts = generate_enhanced_prompts()
    
    # 4. Crear plan de implementación
    implementation_plan = create_implementation_plan(
        best_models, improvements, enhanced_prompts
    )
    
    # Guardar recursos
    output_dir = Path("./improvement_strategy")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "enhanced_prompts.json", "w", encoding="utf-8") as f:
        json.dump(enhanced_prompts, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "improvement_analysis.json", "w", encoding="utf-8") as f:
        json.dump(improvements, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "implementation_plan.json", "w", encoding="utf-8") as f:
        json.dump(implementation_plan, f, indent=2, ensure_ascii=False)
    
    print(f"\nESTRATEGIA GUARDADA EN: {output_dir}/")
    print("\n PRÓXIMOS PASAS RECOMENDADOS:")
    print("1. Ejecutar Phase 1 con Phi-3-mini y prompts mejorados")
    print("2. Comparar resultados con baseline actual") 
    print("3. Iterar basado en resultados")

if __name__ == "__main__":

    main()
