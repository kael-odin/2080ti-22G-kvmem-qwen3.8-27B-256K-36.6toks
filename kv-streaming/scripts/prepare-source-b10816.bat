@echo off
setlocal

set "ROOT=%~dp0..\"
set "SRC=%ROOT%upstream-b10816"
set "REPO=https://github.com/ggml-org/llama.cpp"
set "COMMIT=427291b5b34cd914a31b3fd3b61a68f6184f4b9f"
set "PATCH=%ROOT%patches\0002-merge-adaptive-kv-into-b10816-sm75.patch"

if exist "%SRC%\.git" (
    echo [ERROR] Source tree already exists: %SRC%
    echo Remove it explicitly before preparing a fresh tree.
    exit /b 1
)

set "NO_PROXY=*"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
git -c http.proxy= -c https.proxy= clone "%REPO%" "%SRC%"
if errorlevel 1 exit /b %errorlevel%

git -C "%SRC%" checkout --detach "%COMMIT%"
if errorlevel 1 exit /b %errorlevel%

git -C "%SRC%" apply --check "%PATCH%"
if errorlevel 1 exit /b %errorlevel%
git -C "%SRC%" apply "%PATCH%"
if errorlevel 1 exit /b %errorlevel%

echo Prepared b10816 + adaptive-KV source at %SRC%
exit /b 0
