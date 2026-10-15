@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ---- collect candidate python interpreters ----
set "CAND_PY="
set "CAND_PY3="
set "CAND_LAUNCHER="
for %%P in (python.exe) do if not "%%~$PATH:P"=="" set "CAND_PY=%%~$PATH:P"
for %%P in (python3.exe) do if not "%%~$PATH:P"=="" set "CAND_PY3=%%~$PATH:P"
for %%P in (py.exe) do if not "%%~$PATH:P"=="" set "CAND_LAUNCHER=%%~$PATH:P"
set "CAND_BUNDLED=C:\Users\white\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

rem ---- pick the first interpreter that really has numpy + pandas ----
set "PYEXE="
for %%C in ("%CAND_PY%" "%CAND_LAUNCHER%" "%CAND_PY3%" "%CAND_BUNDLED%") do (
  if not defined PYEXE if not "%%~C"=="" (
    "%%~C" -c "import sys, numpy, pandas" >nul 2>nul
    if not errorlevel 1 set "PYEXE=%%~C"
  )
)

echo.
echo ============================================================
echo   One-click measurement data analysis  -  mda
echo   - double click   : analyze every data file in this folder
echo   - drag a file on : analyze that file only
echo ============================================================

if not defined PYEXE (
  echo.
  echo [ERROR] No usable Python found.
  echo         Need Python 3.10 or newer with numpy and pandas.
  echo         Install with:  pip install numpy pandas openpyxl pillow
  echo.
  pause
  exit /b 1
)
echo   Python: %PYEXE%

set "PYTHONIOENCODING=utf-8"
"%PYEXE%" "analyze.py" --open %*
echo.
pause
