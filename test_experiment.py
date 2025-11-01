#!/usr/bin/env python3
"""
Prueba rápida del entorno OpenVINO
"""
import os
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Configuración básica
OUT_DIR = "./outputs"
Path(OUT_DIR).mkdir(exist_ok=True)

try:
    from transformers import AutoTokenizer
    from optimum.intel.openvino import OVModelForCausalLM
    import openvino as ov
    import torch
    
    print("✅ Todas las dependencias importadas correctamente")
    print(f"📊 OpenVINO version: {ov.__version__}")
    print(f"📊 PyTorch version: {torch.__version__}")
    
    # Verificar dispositivos disponibles
    core = ov.Core()
    devices = core.available_devices
    print(f"🎯 Dispositivos OpenVINO disponibles: {devices}")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    exit(1)

def test_model_loading():
    """Probar carga de un modelo pequeño"""
    print("\n🔍 Probando carga de modelo...")
    
    # Modelo pequeño para prueba rápida
    MODEL_ID = "microsoft/DialoGPT-small"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("🔄 Cargando modelo OpenVINO...")
        ov_model = OVModelForCausalLM.from_pretrained(
            MODEL_ID,
            export=True,
            device="CPU",  # Usar CPU para prueba inicial
            compile=True,
            trust_remote_code=True,
        )
        
        # Prueba de generación
        prompt = "Extrae altura y peso: Paciente de 35 años, altura 1.75 m, peso 80 kg."
        inputs = tokenizer(prompt, return_tensors="pt")
        
        outputs = ov_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Respuesta del modelo: {response}")
        print("✅ ¡Modelo funcionando correctamente!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error con el modelo: {e}")
        return False

def test_data_processing():
    """Probar procesamiento de datos"""
    print("\n📊 Probando procesamiento de datos...")
    
    # Crear datos de ejemplo
    sample_data = {
        'patient_id': ['TEST_001', 'TEST_002'],
        'text': [
            'Paciente masculino, altura 1.80 m, peso 85 kg, BMI 26.2',
            'Mujer de 45 años, sin medidas antropométricas registradas'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("./outputs/test_data.csv", index=False)
    print("✅ Datos de prueba creados en: ./outputs/test_data.csv")
    
    # Cargar y mostrar
    loaded_df = pd.read_csv("./outputs/test_data.csv")
    print(f"📋 Datos cargados: {len(loaded_df)} registros")
    print(loaded_df.head())
    
    return True

if __name__ == "__main__":
    print("🧪 Iniciando pruebas del entorno...")
    
    success1 = test_model_loading()
    success2 = test_data_processing()
    
    if success1 and success2:
        print("\n🎉 ¡Entorno completamente funcional!")
        print("\n📝 Ahora puedes ejecutar tu experimento completo:")
        print("   python run_experiment.py")
    else:
        print("\n⚠️ Algunas pruebas fallaron, pero el entorno está instalado.")