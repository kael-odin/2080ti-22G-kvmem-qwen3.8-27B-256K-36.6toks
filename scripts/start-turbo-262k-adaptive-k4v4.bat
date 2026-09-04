@echo off
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 MODEL.gguf MMPROJ.gguf [PORT]
    exit /b 1
)
if "%~2"=="" (
    echo Usage: %~nx0 MODEL.gguf MMPROJ.gguf [PORT]
    exit /b 1
)

set "MODEL=%~f1"
set "MMPROJ=%~f2"
set "PORT=%~3"
if not defined PORT set "PORT=18080"
set "SERVER=%~dp0..\bin\llama-server-kv.exe"

"%SERVER%" ^
    -m "%MODEL%" ^
    --mmproj "%MMPROJ%" ^
    --alias qwen3.8-turbo-kv-stream ^
    -c 262144 ^
    -ngl 99 ^
    -ctk q4_0 ^
    -ctv q4_0 ^
    --load-mode none ^
    -fa on ^
    --jinja ^
    --reasoning-format deepseek ^
    --cache-ram 0 ^
    -b 512 ^
    -ub 512 ^
    -np 1 ^
    -t 12 ^
    --spec-type draft-mtp ^
    --spec-draft-n-max 2 ^
    --kv-stream-stage-mib 1024 ^
    --image-min-tokens 1024 ^
    --host 127.0.0.1 ^
    --port %PORT%
