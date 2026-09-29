@echo off
setlocal EnableExtensions
cd /d "%~dp0"
rem ============================================================
rem  KVMem source preparation (Windows) - fetch upstream
rem  kvmem-llama.cpp baseline 13b7a15 (v0.16.0-rc1 + 4 commits,
rem  the exact tree the measured 36.6 tok/s run was built from)
rem  plus llama.cpp submodule pin b81c99b, then apply this
rem  repo's sm_75 / Windows patches. Result = complete source.
rem
rem  Pure ASCII on purpose (GBK console safe). No admin needed.
rem
rem  Usage:  prepare-source.bat [dest-dir]     (default kvmem-src)
rem ============================================================
set "DEST=%~1"
if "%DEST%"=="" set "DEST=kvmem-src"
set "PIN=13b7a155c6aaf038a02380b16bd8b8ae7eb760e7"
set "SUBPIN=b81c99b47"
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"

where git >nul 2>&1 || (echo [ERROR] git not found in PATH. & exit /b 1)

if exist "%DEST%\.git" (
    echo [SKIP] "%DEST%" already cloned - reusing it.
) else (
    echo [1/4] Cloning kvmem-llama.cpp with llama.cpp submodule ...
    git clone https://github.com/kvmem/kvmem-llama.cpp.git "%DEST%" || goto :fail
    pushd "%DEST%" || goto :fail
    git submodule update --init llama.cpp || goto :fail
    popd
)

pushd "%DEST%" || goto :fail
echo [2/4] Checking out baseline %PIN% and pinning llama.cpp to %SUBPIN% ...
git checkout %PIN% || goto :fail
git submodule update --init --recursive || goto :fail
pushd llama.cpp || goto :fail
git checkout %SUBPIN% || goto :fail
popd

echo [3/4] Applying sm_75 / Windows patches ...
git apply --whitespace=nowarn "%HERE%\patches\0001-kvmem-parent-sm75-windows.patch" || goto :fail
copy /y "%HERE%\include\kvmem_win_compat.h" kvmem\include\kvmem\ >nul || goto :fail
pushd llama.cpp || goto :fail
git apply --whitespace=nowarn "%HERE%\patches\llama.cpp-b81c99b-sm75-tree-snapshot.patch" || goto :fail
popd

echo [4/4] Installing build script into the source tree ...
copy /y "%HERE%\build-windows-sm75.bat" build-windows-sm75.bat >nul || goto :fail
popd

echo.
echo [OK] Complete source ready in "%DEST%".
echo      Next:  cd %DEST%
echo             build-windows-sm75.bat
echo      CUDA Toolkit 13.2 Update 2 (nvcc 13.2.86) is required.
echo      No-admin redist setup method:
echo      see kvmem\docs\kvmem-windows-port-zh.md section 2.
exit /b 0

:fail
popd 2>nul
echo.
echo [FAIL] See messages above. Fix and re-run this script.
exit /b 1
