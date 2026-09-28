@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem  KVMem - Windows / RTX 2080 Ti (sm_75) build script
rem  Machine-specific. Upstream scripts\build-cuda.sh targets
rem  Linux + sm_120 and will not work here.
rem
rem  Usage:
rem    build-windows-sm75.bat             configure + build
rem    build-windows-sm75.bat configure   configure only
rem    build-windows-sm75.bat clean       remove build dir
rem
rem  This file must stay pure ASCII and use CRLF line endings.
rem  Chinese characters break under the GBK console codepage.
rem  Do not use "::" comments inside parenthesised blocks.
rem ============================================================

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=build"

rem ---------- Visual Studio ----------
set "VCVARS="
for %%V in (
    "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do if not defined VCVARS if exist %%V set "VCVARS=%%~V"
if not defined VCVARS echo [ERROR] vcvars64.bat not found. & exit /b 1

rem ---------- CMake ----------
set "CMAKE="
for %%V in (
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    "C:\Program Files\CMake\bin\cmake.exe"
) do if not defined CMAKE if exist %%V set "CMAKE=%%~V"
if not defined CMAKE echo [ERROR] cmake.exe not found. & exit /b 1

rem ---------- Ninja ----------
set "NINJA="
for %%V in (
    "C:\Python311\Scripts\ninja.exe"
    "C:\Python314\Scripts\ninja.exe"
) do if not defined NINJA if exist %%V set "NINJA=%%~V"

rem ---------- CUDA ----------
rem 1) project-local redist toolkit (.setup\cuda-13.2-86) - no admin needed
rem 2) system toolkits under NVIDIA GPU Computing Toolkit\CUDA\v*
set "SETUP=%ROOT%\.setup"
set "SYSBASE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
set "CUDA_DIR="

for /d %%D in ("%SETUP%\cuda-*") do if not defined CUDA_DIR if exist "%%~fD\bin\nvcc.exe" set "CUDA_DIR=%%~fD"
if not defined CUDA_DIR for %%V in (13.2 13.3 13.4 13.0 12.8) do if not defined CUDA_DIR if exist "%SYSBASE%\v%%V\bin\nvcc.exe" set "CUDA_DIR=%SYSBASE%\v%%V"
if not defined CUDA_DIR echo [ERROR] No CUDA toolkit found. & exit /b 1

rem build-dir tag: sanitise the CUDA folder name
for %%a in ("%CUDA_DIR%") do set "CUDATAG=%%~nxa"
set "CUDATAG=%CUDATAG:cuda-=%"
set "CUDATAG=%CUDATAG:v=%"
set "CUDATAG=%CUDATAG:.=%"
set "CUDATAG=%CUDATAG:-=%"

set "BUILD=%ROOT%\build-cu%CUDATAG%-sm75"

if /i "%ACTION%"=="clean" goto :do_clean
if /i "%ACTION%"=="configure" set "CFGONLY=1"

echo ============================================================
echo   KVMem Windows build  -  arch sm_75 (Turing / 2080 Ti)
echo ------------------------------------------------------------
echo   VS    : %VCVARS%
echo   CMake : %CMAKE%
echo   Ninja : %NINJA%
echo   CUDA  : %CUDA_DIR%
echo   Build : %BUILD%
echo   Action: %ACTION%
echo ============================================================
echo.

call "%VCVARS%" >nul 2>&1
if errorlevel 1 echo [ERROR] vcvars64.bat failed. & exit /b 1

if defined NINJA set "CMAKEFLAGS=-DCMAKE_MAKE_PROGRAM=%NINJA%"

echo [1/2] Configuring...
"%CMAKE%" -S "%ROOT%" -B "%BUILD%" -G Ninja ^
 -DCMAKE_BUILD_TYPE=Release ^
 %CMAKEFLAGS% ^
 -DCMAKE_CUDA_COMPILER="%CUDA_DIR%\bin\nvcc.exe" ^
 -DCMAKE_CUDA_ARCHITECTURES=75 ^
 -DCMAKE_CXX_FLAGS=/utf-8 ^
 -DCMAKE_C_FLAGS=/utf-8 ^
 -DGGML_CUDA=ON ^
 -DGGML_CUDA_FA_ALL_QUANTS=ON ^
 -DKVMEM_BUILD_LLAMA=ON ^
 -DLLAMA_KVMEM=ON ^
 -DLLAMA_KVMEM_ROOT="%ROOT%"
if errorlevel 1 echo [ERROR] Configure failed. & exit /b 1

if defined CFGONLY goto :do_done

rem ACTION=build -> full build. Any other value -> build just that target.
set "TARGETARG="
if /i not "%ACTION%"=="build" set "TARGETARG=--target %ACTION%"

echo.
echo [2/2] Building (target: %ACTION%). This takes a while...
"%CMAKE%" --build "%BUILD%" %TARGETARG% -j %NUMBER_OF_PROCESSORS%
if not errorlevel 1 goto :build_ok
echo.
echo [ERROR] Build failed. See messages above.
exit /b 1

:build_ok
echo.
echo ============================================================
echo   [OK] Build finished.
echo   Server: %BUILD%\bin\llama-kvmem-server.exe
echo.
echo   WARNING: a successful build does NOT prove IQ3 inference
echo   is correct. Read the output yourself and check for garbage.
echo ============================================================
goto :eof

:do_clean
if exist "%BUILD%" rmdir /s /q "%BUILD%"
echo [OK] removed %BUILD%
goto :eof

:do_done
echo.
echo [OK] Configure succeeded. Build dir: %BUILD%
goto :eof
