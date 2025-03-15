@echo off
echo Running Video Subtitles Adder...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run the application
call python -m src.main %*

:: Check for errors
if %errorlevel% neq 0 (
    echo.
    echo An error occurred while running the application.
    echo Please check the logs for more information.
)
