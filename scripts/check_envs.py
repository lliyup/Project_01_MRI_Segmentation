import sys
import torch

print("=" * 50)
print("Environment Check")
print("=" * 50)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
else:
    print("GPU: CPU only")

packages = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "skimage",
    "nibabel",
    "SimpleITK",
    "tqdm",
]

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, "__version__", "version unknown")
        print(f"{pkg}: {version}")
    except Exception as e:
        print(f"{pkg}: import failed -> {e}")

print("=" * 50)
print("Check finished.")