@echo off
title Integrated Plant Suite - Mobile & Server Launcher

echo ========================================================
echo   Starting Integrated Plant Suite App
echo ========================================================
echo.

REM Get local IPv4 address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip
set LOCAL_IP=%LOCAL_IP: =%

echo [INFO] Local Server URL:  http://localhost:8501
echo [INFO] Mobile Access URL: http://%LOCAL_IP%:8501
echo.
echo [NOTE] Make sure your mobile phone is connected to the same Wi-Fi network!
echo.
echo Launching Streamlit server...
echo ========================================================

streamlit run app.py --server.address=0.0.0.0 --server.port=8501
pause
