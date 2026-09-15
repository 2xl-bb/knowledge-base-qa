@echo off
rem KB-QA launcher: start server if not running, then open the web page
netstat -ano | findstr /r ":8501 .*LISTENING" >nul 2>&1
if errorlevel 1 goto startserver
start http://localhost:8501
exit /b
:startserver
cd /d "D:\L\knowledge-base-qa"
echo Starting KB-QA server... close this window to stop it
start "KB-QA Server" cmd /k "python -m streamlit run app.py"
