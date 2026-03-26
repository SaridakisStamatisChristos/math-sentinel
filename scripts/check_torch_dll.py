import ctypes
import os
import site
import sys
from pathlib import Path


def candidate_site_packages() -> list[Path]:
    candidates: list[Path] = []
    for raw_path in site.getsitepackages():
        candidates.append(Path(raw_path))
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site))
    candidates.append(Path(sys.prefix) / "Lib" / "site-packages")

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def find_torch_python_dll() -> Path | None:
    for site_packages in candidate_site_packages():
        dll_path = site_packages / "torch" / "lib" / "torch_python.dll"
        if dll_path.exists():
            return dll_path.resolve()
    return None


print("python", sys.executable)
print("sys.prefix", sys.prefix)

dll_path = find_torch_python_dll()
print("candidate_site_packages", [str(path) for path in candidate_site_packages()])

if dll_path is None:
    print("DLL_NOT_FOUND")
    raise SystemExit(1)

print("DLL", dll_path)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(dll_path.parent))

try:
    ctypes.WinDLL(str(dll_path))
    print("DLL_LOAD_OK")
except OSError as error:
    print("DLL_LOAD_ERROR", error)
    print("GetLastError", ctypes.GetLastError())
    raise
