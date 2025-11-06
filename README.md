# Resumen del TFM

## Título del proyecto
**Estudio de la mejora del uso de LLMs en cálculos médicos**

## Objetivo general
Evaluar y mejorar la exactitud y robustez de los *Large Language Models (LLMs)* en cálculos médicos sin aplicar reentrenamiento, empleando estrategias de *prompt engineering* y validación estructurada.



## 1. Enfoque general

El proyecto aborda un problema poco explorado: la capacidad de los LLMs para **razonar numéricamente en contexto clínico**, por ejemplo, calcular el IMC a partir de texto libre.  
Se diseña un **pipeline reproducible de evaluación**, centrado en la extracción, normalización y verificación de variables antropométricas (altura, peso e IMC).



## 2. Conjunto de datos

- **Fuente:** *PMC-Patients dataset* (Zhao et al., 2023), con 167.034 registros de pacientes descritos en artículos médicos de PubMed Central.  
- **Subconjunto utilizado:** 3.811 pacientes con valores plausibles y completos de altura, peso e IMC.

**Archivos principales:**
- `eval_imc_fullnotes.csv` → notas clínicas completas (entrada a los modelos).  
- `valid_imc.csv` → *ground truth* validado manualmente (referencia de evaluación).  
- `pred_*.csv` → predicciones generadas por los modelos.  
- `eval_*.csv` → comparativa entre predicciones y *ground truth* (MAE, RMSE, precisión).

---

## 3. Pipeline experimental

El pipeline automatiza todas las etapas desde la carga de datos hasta el cálculo de métricas, garantizando trazabilidad.


**Entorno:**
- Ejecutado localmente en VS Code, con **OpenVINO** sobre GPU Intel Iris.  
- Permite exportar modelos en formato IR para acelerar la inferencia y mantener reproducibilidad.



## 4. Modelos evaluados

Cinco modelos representando distintas arquitecturas y tamaños:

| Modelo | Parámetros | Propósito |
|---------|-------------|------------|
| Qwen2.5-0.5B-Instruct | 0.5B | Base ligera multilingüe. |
| StableLM-Zephyr-3B | 3B | Generalista, ajustado por diálogo instructivo. |
| Phi-3-mini-4k-instruct | 3.8B | Equilibrio entre rendimiento y coste. |
| CodeLlama-7B-Instruct-hf | 7B | Generación estructurada y razonamiento lógico. |
| DialoGPT-large | 774M | Arquitectura dialógica para evaluar transferencia. |



## 5. Estrategias de prompting

Se definieron cuatro variantes aplicadas a todos los modelos:

| Versión | Tipo | Descripción |
|----------|------|-------------|
| v1_simple | Directa | Extracción básica de {altura, peso, IMC}. |
| v2_estricto | Con reglas | Validación de unidades y rangos fisiológicos. |
| v3_fewshot | Ejemplos guía | Añade tres ejemplos de entrada/salida JSON. |
| v4_encadenado | Dos pasos | Razonamiento por fases (detección + normalización). |



## 6. Evaluación y métricas

Se utilizó una **taxonomía de errores** inspirada en *MedCalc-Bench* (Khandekar et al., 2024):

- **A – Knowledge:** omisión de información explícita.  
- **B – Extraction:** valor erróneo o fuera de rango.  
- **C – Arithmetic:** error en la fórmula o cálculo.  

**Métricas cuantitativas:**
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- Precisión total y por variable  



## 7. Resultados

- La precisión depende fuertemente del diseño del prompt.  
- El modelo **Phi-3-mini-4k-instruct** alcanzó hasta **95 % de exactitud** bajo instrucciones estructuradas.  
- Los errores más comunes fueron de tipo **A (omisiones)**.  
- Las estrategias **v1_simple** y **v2_estricto** lograron el mejor equilibrio entre coherencia y cobertura.  



## 8. Histórico de pruebas y limitaciones

Inicialmente se exploraron técnicas de *fine-tuning* (QLoRA y LoRA) sobre modelos como *Mistral-7B* y *BioGPT*, pero se descartaron por **limitaciones de hardware** (GPU y memoria).  
Esto llevó a consolidar una metodología basada únicamente en *prompting estructurado*, que resultó más eficiente, reproducible y viable localmente.



## 9. Conclusiones

- Se demostró la viabilidad del uso de LLMs en cálculos médicos sin reentrenamiento, mediante un pipeline reproducible y métricas controladas.  
- La ingeniería de *prompts* tiene un **impacto directo en la precisión**.  
- Se propone un esquema experimental estandarizado (*MedCalc-Bench adaptado*) para evaluar razonamiento clínico en modelos generativos.  

**Líneas futuras:**
- Integrar verificadores externos (NumPy, reglas fisiológicas).  
- Explorar *chain-of-thought* clínico.  
- Optimizar despliegues locales con cuantización y compresión.



## 10. Reflexión final

El proyecto evidencia que, incluso con recursos limitados, es posible generar investigación empírica sólida y reproducible aplicando buenas prácticas de ingeniería de datos y evaluación.  
Además, plantea una base metodológica para futuros trabajos que combinen **LLMs + razonamiento médico estructurado**.
