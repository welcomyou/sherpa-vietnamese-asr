"""Script chan doan loi DLL cho ban portable. Chay bang:
  python\python.exe diagnose_dll.py
"""
import sys, os, ctypes

print("=" * 50)
print("Chan doan DLL - Sherpa Vietnamese ASR")
print("=" * 50)

python_dir = os.path.dirname(sys.executable)
base_dir = os.path.dirname(python_dir)
sp = os.path.join(python_dir, 'Lib', 'site-packages')

print(f"\nPython: {sys.executable}")
print(f"Version: {sys.version}")
print(f"Platform: {sys.platform}")

# Check critical directories
dirs_to_check = {
    "PyQt6/Qt6/bin": os.path.join(sp, 'PyQt6', 'Qt6', 'bin'),
    "PyQt6/Qt6/plugins/platforms": os.path.join(sp, 'PyQt6', 'Qt6', 'plugins', 'platforms'),
    "sherpa_onnx/lib": os.path.join(sp, 'sherpa_onnx', 'lib'),
    "onnxruntime/capi": os.path.join(sp, 'onnxruntime', 'capi'),
    "numpy/_core": os.path.join(sp, 'numpy', '_core'),
}

print("\n--- Thu muc ---")
for name, path in dirs_to_check.items():
    exists = os.path.exists(path)
    print(f"  {'OK' if exists else 'THIEU'}: {name}")
    if exists and name == "PyQt6/Qt6/bin":
        dlls = [f for f in os.listdir(path) if f.endswith('.dll')]
        print(f"       DLLs: {len(dlls)} files")
        for d in sorted(dlls):
            print(f"         {d}")

# Check VC++ Runtime
print("\n--- VC++ Runtime ---")
vcrt_dlls = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'concrt140.dll']
for dll in vcrt_dlls:
    # Check in python dir
    in_python = os.path.exists(os.path.join(python_dir, dll))
    # Check system
    try:
        ctypes.WinDLL(dll)
        in_system = True
    except OSError:
        in_system = False
    status = "OK" if (in_python or in_system) else "THIEU"
    where = []
    if in_python: where.append("portable")
    if in_system: where.append("system")
    print(f"  {status}: {dll} ({', '.join(where)})")

# Check UCRT (api-ms-win-crt)
print("\n--- Universal CRT ---")
ucrt_dlls = [
    'api-ms-win-crt-runtime-l1-1-0.dll',
    'api-ms-win-crt-stdio-l1-1-0.dll',
    'api-ms-win-crt-heap-l1-1-0.dll',
    'api-ms-win-crt-math-l1-1-0.dll',
]
for dll in ucrt_dlls:
    try:
        ctypes.WinDLL(dll)
        print(f"  OK: {dll}")
    except OSError:
        print(f"  THIEU: {dll} *** CAN CAI Windows Update KB2999226 ***")

# Check PATH
print("\n--- PATH co PyQt6 ---")
qt_in_path = any('PyQt6' in p for p in os.environ.get('PATH', '').split(os.pathsep))
print(f"  PyQt6/Qt6/bin in PATH: {qt_in_path}")

# Try loading Qt6 DLLs manually
print("\n--- Thu load DLL ---")
qt_bin = os.path.join(sp, 'PyQt6', 'Qt6', 'bin')
for dll_name in ['Qt6Core.dll', 'Qt6Gui.dll', 'Qt6Widgets.dll']:
    dll_path = os.path.join(qt_bin, dll_name)
    if not os.path.exists(dll_path):
        print(f"  THIEU FILE: {dll_name}")
        continue
    try:
        ctypes.WinDLL(dll_path)
        print(f"  OK: {dll_name}")
    except OSError as e:
        print(f"  LOI: {dll_name} - {e}")

# Try importing PyQt6
print("\n--- Thu import PyQt6 ---")
try:
    from PyQt6 import sip
    print("  OK: PyQt6.sip")
except ImportError as e:
    print(f"  LOI: PyQt6.sip - {e}")

try:
    from PyQt6.QtCore import QCoreApplication
    print("  OK: PyQt6.QtCore")
except ImportError as e:
    print(f"  LOI: PyQt6.QtCore - {e}")

try:
    from PyQt6.QtWidgets import QApplication
    print("  OK: PyQt6.QtWidgets")
except ImportError as e:
    print(f"  LOI: PyQt6.QtWidgets - {e}")

print("\n" + "=" * 50)
print("Xong. Gui ket qua nay de duoc ho tro.")
print("=" * 50)
input("Nhan Enter de dong...")
