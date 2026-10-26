@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Pick a Python >= 3.11: TSG_PYTHON > py -3.13/3.12/3.11 > python
set "PY="
if defined TSG_PYTHON set "PY=%TSG_PYTHON%"
if not defined PY (
  for %%v in (3.13 3.12 3.11) do (
    if not defined PY (
      py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
    )
  )
)
if not defined PY (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.11+ was not found.
  echo.
  echo Option 1: install Python 3.11 or newer, then reopen the terminal.
  echo Option 2: set TSG_PYTHON to an existing interpreter, e.g.
  echo            set TSG_PYTHON=C:\Python312\python.exe
  echo.
  pause
  exit /b 1
)

echo Interpreter: %PY%
%PY% scripts\register_demo.py
if errorlevel 1 goto fail

echo.
echo Dashboard: http://127.0.0.1:8765
if not defined TSG_NO_BROWSER start "" http://127.0.0.1:8765
%PY% -m monitoring.server --port 8765
goto :eof

:fail
echo.
echo Startup failed. Run: %PY% scripts\check_env.py
pause
exit /b 1
