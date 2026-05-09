@echo off
:: ============================================================
:: deploy_hf.bat  —  Deploy LevelUp AI to HuggingFace Spaces
:: Run from the project root: deploy_hf.bat YOUR_HF_USERNAME
:: ============================================================

IF "%1"=="" (
    echo Usage: deploy_hf.bat YOUR_HF_USERNAME
    exit /b 1
)

SET USERNAME=%1
SET ADAPTER_REPO=%USERNAME%/levelup-qlora

echo.
echo [1/4] Uploading LoRA adapter to HuggingFace Hub...
huggingface-cli upload %ADAPTER_REPO% models/final/levelup-qlora-cloud . --repo-type model
IF ERRORLEVEL 1 (
    echo WARNING: Adapter upload failed. Make sure models/final/levelup-qlora-cloud exists.
    echo          You can upload it manually at https://huggingface.co/new
)

echo.
echo [2/4] Creating and pushing FRONTEND Space...
IF NOT EXIST ".hf_frontend" mkdir .hf_frontend
cd .hf_frontend
git clone https://huggingface.co/spaces/%USERNAME%/levelup-ai . 2>nul || (
    git init
    git remote add origin https://huggingface.co/spaces/%USERNAME%/levelup-ai
)
copy /Y ..\spaces\frontend\README.md README.md
copy /Y ..\spaces\frontend\index.html index.html
git add README.md index.html
git commit -m "Deploy LevelUp AI frontend" --allow-empty
git push origin main
cd ..

echo.
echo [3/4] Creating and pushing BACKEND Space...
IF NOT EXIST ".hf_backend" mkdir .hf_backend
cd .hf_backend
git clone https://huggingface.co/spaces/%USERNAME%/levelup-ai-api . 2>nul || (
    git init
    git remote add origin https://huggingface.co/spaces/%USERNAME%/levelup-ai-api
)
:: Copy backend app files
copy /Y ..\spaces\backend\README.md  README.md
copy /Y ..\spaces\backend\app.py     app.py
copy /Y ..\spaces\backend\requirements.txt requirements.txt

:: Copy project source files needed by the backend
IF NOT EXIST chatbot mkdir chatbot
IF NOT EXIST voice    mkdir voice
IF NOT EXIST anomaly_detection mkdir anomaly_detection
xcopy /E /I /Y ..\chatbot       chatbot
xcopy /E /I /Y ..\voice         voice
xcopy /E /I /Y ..\anomaly_detection anomaly_detection

git add .
git commit -m "Deploy LevelUp AI backend" --allow-empty
git push origin main
cd ..

echo.
echo [4/4] Done!
echo.
echo   Frontend : https://huggingface.co/spaces/%USERNAME%/levelup-ai
echo   Backend  : https://huggingface.co/spaces/%USERNAME%/levelup-ai-api
echo   Adapter  : https://huggingface.co/models/%USERNAME%/levelup-qlora
echo.
echo NEXT STEPS:
echo   1. Go to Backend Space Settings ^> Variables and Secrets
echo   2. Add: HF_TOKEN, LEVELUP_ADAPTER=%ADAPTER_REPO%
echo   3. Add: ELEVENLABS_API_KEY (optional)
echo   4. Apply for ZeroGPU at: https://huggingface.co/zero-gpu-explorers
echo.
