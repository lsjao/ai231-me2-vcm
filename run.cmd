@echo off
rem Runs a project module from src/ with the Python that has the project's packages installed,
rem regardless of which python/venv your terminal defaults to. Usage:
rem   run.cmd vcm.record_dataset --speaker josh --condition quiet --distance near --auto
rem   run.cmd vcm.train --data-root .. --manifest manifest.csv --output-dir ../models --extra-data ../data_real
cd /d "%~dp0src"
"C:\Users\Josh\AppData\Local\Programs\Python\Python313\python.exe" -m %*
