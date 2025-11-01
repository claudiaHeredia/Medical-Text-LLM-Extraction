# Reporte Comparativo Completo - MEDCALC-Bench

## 📊 Resumen Ejecutivo

**Mejor configuración general**: Phi-3-mini-4k-instruct - v2_estricto (95.0% accuracy)

## 🏆 Top 5 Configuraciones

| Modelo                 | Estrategia    |   predijo_bien_all |   predijo_bien_bmi |
|:-----------------------|:--------------|-------------------:|-------------------:|
| Phi-3-mini-4k-instruct | v2_estricto   |                 95 |                 95 |
| Phi-3-mini-4k-instruct | v1_simple     |                 90 |                 90 |
| Phi-3-mini-4k-instruct | v3_fewshot    |                 90 |                 90 |
| Phi-3-mini-4k-instruct | v4_encadenado |                 40 |                 40 |
| stablelm-zephyr-3b     | v2_estricto   |                 40 |                 40 |

## 🔝 Mejor Estrategia por Modelo

| Modelo                   | Estrategia   |   predijo_bien_all |   predijo_bien_bmi |
|:-------------------------|:-------------|-------------------:|-------------------:|
| Phi-3-mini-4k-instruct   | v2_estricto  |                 95 |                 95 |
| stablelm-zephyr-3b       | v2_estricto  |                 40 |                 40 |
| Qwen2.5-0.5B-Instruct    | v2_estricto  |                  5 |                 10 |
| CodeLlama-7b-Instruct-hf | v1_simple    |                  0 |                  0 |
| DialoGPT-large           | v1_simple    |                  0 |                  0 |

## 📈 Comparativa de Estrategias (Promedio)

| Estrategia    |   predijo_bien_all_mean |   predijo_bien_all_std |   predijo_bien_all_count |   predijo_bien_bmi_mean |   error_A_knowledge_mean |   error_B_extraction_mean |   error_C_arithmetic_mean |
|:--------------|------------------------:|-----------------------:|-------------------------:|------------------------:|-------------------------:|--------------------------:|--------------------------:|
| v1_simple     |                      22 |                   39   |                        5 |                      26 |                       73 |                         6 |                         1 |
| v2_estricto   |                      28 |                   41   |                        5 |                      29 |                       61 |                        13 |                         1 |
| v3_fewshot    |                      22 |                   39   |                        5 |                      26 |                       74 |                         6 |                         0 |
| v4_encadenado |                       8 |                   17.9 |                        5 |                       9 |                       71 |                        16 |                        11 |

## 🔍 Análisis de Errores

- **Error A (Conocimiento)**: 69.8% promedio
- **Error B (Extracción)**: 10.2% promedio
- **Error C (Aritmético)**: 3.2% promedio

## 💡 Recomendaciones

1. **Configuración recomendada**: Phi-3-mini-4k-instruct con estrategia v2_estricto
2. **Estrategia más consistente**: v4_encadenado (menor variabilidad entre modelos)
3. **Modelo más versátil**: Phi-3-mini-4k-instruct (mejor rendimiento promedio across estrategias)
