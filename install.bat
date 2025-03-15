echo Installing Video Subtitles Adder...

:: Check if Python is installed
call python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    call python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
call pip install -r requirements.txt

:: Check if FFmpeg is installed
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: FFmpeg is not installed or not in PATH.
    echo Please install FFmpeg and add it to your PATH.
    echo You can download FFmpeg from https://ffmpeg.org/download.html
)

echo.
echo Installation completed!
echo To run the application, use run.bat
echo.

pause
