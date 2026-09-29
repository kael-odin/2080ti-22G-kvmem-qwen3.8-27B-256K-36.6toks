@echo off
setlocal

set "ROOT=%~dp0..\"
set "SRC=%ROOT%upstream"
set "REPO=https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming.git"
set "COMMIT=d873e5db9a698a8063347e586ed047242d51fdce"
set "PATCH=%ROOT%patches\0001-cuda-fix-adaptive-kv-streaming-on-sm75.patch"

if exist "%SRC%\.git" (
    echo [ERROR] Source tree already exists: %SRC%
    echo Remove it explicitly before preparing a fresh tree.
    exit /b 1
)

set "NO_PROXY=*"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
git -c http.proxy= -c https.proxy= clone --branch feature/adaptive-kv-stream --single-branch "%REPO%" "%SRC%"
if errorlevel 1 exit /b %errorlevel%

git -C "%SRC%" checkout --detach "%COMMIT%"
if errorlevel 1 exit /b %errorlevel%

git -C "%SRC%" apply --check "%PATCH%"
if errorlevel 1 exit /b %errorlevel%
git -C "%SRC%" apply "%PATCH%"
if errorlevel 1 exit /b %errorlevel%

echo Prepared adaptive KV source at %SRC%
exit /b 0
