import importlib.util
import platform

REQUIRED = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("accelerate", "Accelerate"),
    ("bitsandbytes", "BitsAndBytes"),
    ("qwen_vl_utils", "qwen-vl-utils"),
    ("PIL", "Pillow"),
    ("safetensors", "safetensors"),
]

print("Python:", platform.python_version())
print("Platform:", platform.platform())
missing=[]
for module,label in REQUIRED:
    ok=importlib.util.find_spec(module) is not None
    print(f"{label:18s}: {'OK' if ok else 'MISSING'}")
    if not ok: missing.append(label)

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
except Exception as exc:
    print("Torch/CUDA check failed:", exc)

if missing:
    raise SystemExit("Missing VLM dependencies. Install requirements-vlm-rtx4060.txt")
print("Environment preflight: PASS")
