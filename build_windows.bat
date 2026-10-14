@echo off
REM Local build on a Windows PC (needs Python 3.11+ and Inno Setup 6 installed once).
REM Result: installer_output\MHous3-Audio-Studio-Setup.exe
setlocal
python -m pip install -r requirements.txt pyinstaller || goto :err
if not exist bin mkdir bin
if not exist bin\ffmpeg.exe (
  powershell -NoProfile -Command "Invoke-WebRequest 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip' -OutFile ff.zip; Expand-Archive ff.zip ffx -Force; Copy-Item (Get-ChildItem ffx -Recurse -Filter ffmpeg.exe | Select-Object -First 1).FullName bin\ffmpeg.exe"
)
if not exist bin\deep-filter.exe (
  powershell -NoProfile -Command "Invoke-WebRequest 'https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-pc-windows-msvc.exe' -OutFile bin\deep-filter.exe"
)
pyinstaller --noconfirm MHous3.spec || goto :err
"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" installer\mhous3.iss || goto :err
echo.
echo DONE - your installer is in installer_output\
pause
exit /b 0
:err
echo BUILD FAILED - see messages above.
pause
exit /b 1
