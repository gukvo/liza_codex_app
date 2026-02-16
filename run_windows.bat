@echo off
setlocal

rem Run from Windows PowerShell or double-click this file.
rem This script expects Python 3.11 and a working MediaPipe FaceMesh install.

set "ROOT=%~dp0"
set "VER=3.11"
set "PY=py -%VER%"

%PY% -V >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 not found.
  echo Install Python 3.11 x64 and try again.
  exit /b 2
)

set "VENV=%LOCALAPPDATA%\liza_codex_app\venv311"
if not exist "%VENV%\Scripts\python.exe" (
  %PY% -m venv "%VENV%"
)

set "VENV_PY=%VENV%\Scripts\python.exe"

%VENV_PY% -m ensurepip --upgrade >nul 2>&1
%VENV_PY% -m pip install --upgrade pip
%VENV_PY% -m pip install --upgrade ^
  "numpy==1.26.4" ^
  "opencv-python==4.8.1.78" ^
  "opencv-contrib-python==4.8.1.78" ^
  "mediapipe==0.10.21" ^
  "onnxruntime>=1.18,<2" ^
  "norfair>=2.2,<3"

echo Using Python %VER% from %VENV_PY%
%VENV_PY% -c "import sys; print(sys.version)"

set "CHECK=%TEMP%\mp_face_mesh_check_%RANDOM%.py"
> "%CHECK%" echo import importlib, traceback
>> "%CHECK%" echo ok = False
>> "%CHECK%" echo err = None
>> "%CHECK%" echo try:
>> "%CHECK%" echo     importlib.import_module("mediapipe.solutions.face_mesh")
>> "%CHECK%" echo     ok = True
>> "%CHECK%" echo except Exception as exc:
>> "%CHECK%" echo     err = exc
>> "%CHECK%" echo if not ok:
>> "%CHECK%" echo     try:
>> "%CHECK%" echo         importlib.import_module("mediapipe.python.solutions.face_mesh")
>> "%CHECK%" echo         ok = True
>> "%CHECK%" echo     except Exception as exc:
>> "%CHECK%" echo         err = exc
>> "%CHECK%" echo if not ok:
>> "%CHECK%" echo     print("FaceMesh import failed:", err)
>> "%CHECK%" echo     traceback.print_exc()
>> "%CHECK%" echo     raise SystemExit(1)

%VENV_PY% "%CHECK%" >nul 2>nul
set "CHECK_ERR=%errorlevel%"
del /f /q "%CHECK%" >nul 2>&1
if not "%CHECK_ERR%"=="0" (
  echo.
  echo FaceMesh init check failed.
  echo Install Microsoft Visual C++ 2015-2022 Redistributable x64, then retry.
  exit /b 3
)

%VENV_PY% "%ROOT%eye_state_cam.py"
endlocal
