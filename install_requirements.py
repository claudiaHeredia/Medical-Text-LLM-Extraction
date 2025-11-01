#!/usr/bin/env python3
"""
Instalación directa de requerimientos para OpenVINO
"""
import subprocess
import sys
import os

def install_requirements():
    requirements = [
        "numpy<2.1",
        "pandas",
        "tqdm", 
        "scikit-learn",
        "transformers==4.46.2",
        "accelerate>=0.34",
        "huggingface_hub>=0.24",
        "sacremoses",
        "sentencepiece", 
        "optimum-intel[openvino]==1.26.0",
        "openvino>=2025.1.0",
        "torch",
        "torchvision",
        "torchaudio"
    ]
    
    print("🚀 Instalando dependencias...")
    for package in requirements:
        print(f"📦 Instalando {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} instalado")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {package}: {e}")
    
    print("\n🔍 Verificando instalación...")
    try:
        import openvino as ov
        from transformers import AutoTokenizer
        from optimum.intel.openvino import OVModelForCausalLM
        import torch
        print("✅ Todas las dependencias instaladas correctamente!")
        print(f"✅ OpenVINO version: {ov.__version__}")
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ Error en verificación: {e}")
        return False
    
    return True

if __name__ == "__main__":
    install_requirements()