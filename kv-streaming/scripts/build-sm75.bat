@echo off
setlocal

set "ROOT=%~dp0..\"
if not defined SRC set "SRC=%ROOT%upstream"
if not defined BUILD set "BUILD=%ROOT%build"
if not defined CMAKE set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined NVCC set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"

if not exist "%SRC%\CMakeLists.txt" (
    echo [ERROR] Missing source tree: %SRC%
    echo Run scripts\prepare-source.bat first or set SRC.
    exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b %errorlevel%

"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G "Ninja Multi-Config" ^
    -DCMAKE_CUDA_COMPILER="%NVCC%" ^
    -DCMAKE_CUDA_ARCHITECTURES=75 ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DGGML_CUDA_CUB_3DOT2=OFF ^
    -DGGML_CUDA_NCCL=OFF ^
    -DGGML_BACKEND_DL=OFF ^
    -DGGML_NATIVE=OFF ^
    -DLLAMA_OPENSSL=OFF ^
    -DLLAMA_BUILD_UI=OFF ^
    -DLLAMA_USE_PREBUILT_UI=OFF ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_BUILD_APP=OFF ^
    -DLLAMA_BUILD_SERVER=ON ^
    -DLLAMA_BUILD_TOOLS=ON ^
    -DLLAMA_BUILD_COMMON=ON
if errorlevel 1 exit /b %errorlevel%

"%CMAKE%" --build "%BUILD%" --config Release --target llama-server llama-perplexity --parallel 6
exit /b %errorlevel%
