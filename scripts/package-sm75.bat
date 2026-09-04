@echo off
setlocal

set "ROOT=%~dp0..\"
if not defined BUILD set "BUILD=%ROOT%build"
if not defined CUDA_BIN set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
set "SOURCE=%BUILD%\bin\Release"
set "DEST=%ROOT%bin"

if not exist "%SOURCE%\llama-server.exe" (
    echo [ERROR] Missing build output: %SOURCE%\llama-server.exe
    exit /b 1
)

if not exist "%DEST%" mkdir "%DEST%"
copy /Y "%SOURCE%\*.dll" "%DEST%\" >nul
copy /Y "%SOURCE%\llama-server.exe" "%DEST%\llama-server-kv.exe" >nul
if exist "%SOURCE%\llama-perplexity.exe" copy /Y "%SOURCE%\llama-perplexity.exe" "%DEST%\" >nul
if exist "%SOURCE%\llama-perplexity-impl.dll" copy /Y "%SOURCE%\llama-perplexity-impl.dll" "%DEST%\" >nul
copy /Y "%CUDA_BIN%\cudart64_12.dll" "%DEST%\" >nul
copy /Y "%CUDA_BIN%\cublas64_12.dll" "%DEST%\" >nul
copy /Y "%CUDA_BIN%\cublasLt64_12.dll" "%DEST%\" >nul

echo Packaged binaries in %DEST%
exit /b 0
