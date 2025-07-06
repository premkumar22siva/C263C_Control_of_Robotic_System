@echo off

REM Check if the venv folder exists
IF EXIST "venv" (
    call venv\Scripts\activate.bat
) ELSE (
    python3.10.exe -m venv venv
    call venv\Scripts\activate.bat
    python3.10.exe -m pip install --upgrade pip setuptools
    python3.10.exe -m pip install pyyaml --no-warn-script-location
    python3.10.exe -m pip install ./mechae263C_helpers --no-warn-script-location
    python3.10.exe -m pip install ./dynamixel-controller --no-warn-script-location
)