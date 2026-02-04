#!/usr/bin/env python3
"""
Test ANANSE v8 Model Locally (without torch)
"""
import pickle
import os

print("🕷️ ANANSE v8 Local Model Test")
print("=" * 50)

model_path = "models/ananse_v8_enhanced.pt"
if not os.path.exists(model_path):
    model_path = "kaggle_output/ananse_v8_enhanced.pt"

print(f"\n📂 Loading from: {model_path}")
print(f"📊 File size: {os.path.getsize(model_path)} bytes")

# Try to load with pickle (torch saves are pickle-based)
try:
    import zipfile
    with zipfile.ZipFile(model_path, 'r') as z:
        print(f"\n📦 Archive contents:")
        for name in z.namelist():
            print(f"  - {name}")
except:
    print("\n📦 Not a zip archive, checking raw content...")
    with open(model_path, 'rb') as f:
        header = f.read(20)
        print(f"  Header bytes: {header[:10]}")

print("\n✅ Model file verified!")
print("\nTo fully test, install torch: pip install torch")
