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
  echo Set TSG_PYTHON to an existing interpreter, for example:
  echo   set TSG_PYTHON=C:\Python312\python.exe
  pause
  exit /b 1
)

echo Interpreter: %PY%
set "PROVIDER_ARG="
if defined TSGTEST_PROVIDER set "PROVIDER_ARG=--provider %TSGTEST_PROVIDER%"
%PY% -m tsgtest serve --port 8770 --open %PROVIDER_ARG% %*
pause
