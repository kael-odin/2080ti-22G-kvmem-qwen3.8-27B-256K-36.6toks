@echo off
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 MODEL.gguf WIKI.TEST.RAW OUTPUT.LOG
    exit /b 1
)
if "%~2"=="" (
    echo Usage: %~nx0 MODEL.gguf WIKI.TEST.RAW OUTPUT.LOG
    exit /b 1
)
if "%~3"=="" (
    echo Usage: %~nx0 MODEL.gguf WIKI.TEST.RAW OUTPUT.LOG
    exit /b 1
)

set "MODEL=%~f1"
set "DATA=%~f2"
set "OUTPUT=%~f3"
set "PPL=%~dp0..\bin\llama-perplexity.exe"

"%PPL%" ^
    -m "%MODEL%" ^
    -f "%DATA%" ^
    -c 1024 ^
    -b 512 ^
    -ub 512 ^
    -ngl 99 ^
    -fa on ^
    -ctk q8_0 ^
    -ctv q8_0 ^
    --chunks 96 ^
    --load-mode none > "%OUTPUT%" 2>&1

exit /b %errorlevel%
