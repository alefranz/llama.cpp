@echo off
setlocal EnableDelayedExpansion

set "BUILD_DIR=%~1"
if not defined BUILD_DIR set "BUILD_DIR=build-cuda-fa-all-quants"

set "CONFIG=%~2"
if not defined CONFIG set "CONFIG=Release"

set "INSTALL_DIR=%~3"
if not defined INSTALL_DIR set "INSTALL_DIR=C:\Software\llama\current"

pushd "%~dp0.."
if errorlevel 1 goto ERROR

set "SOURCE=%BUILD_DIR%\bin\%CONFIG%"
if not exist "%SOURCE%" (
    echo Error: Build output directory not found: %SOURCE%
    echo Run build-cuda-fa-all-quants.cmd first.
    popd
    exit /B 1
)

echo Installing to %INSTALL_DIR% ...
if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
    if errorlevel 1 goto ERROR
)

robocopy "%SOURCE%" "%INSTALL_DIR%" /E /IS /IT /PURGE /NFL /NDL /NJH /NJS
if errorlevel 8 goto ERROR

if defined CUDA_PATH (
    set "CUDA_DLLS=!CUDA_PATH!\bin\x64"
    if exist "!CUDA_DLLS!" (
        echo Copying CUDA runtime DLLs from !CUDA_DLLS! ...
        copy /Y "!CUDA_DLLS!\cublas64_13.dll" "%INSTALL_DIR%\" >nul
        copy /Y "!CUDA_DLLS!\cublasLt64_13.dll" "%INSTALL_DIR%\" >nul
        copy /Y "!CUDA_DLLS!\cudart64_13.dll" "%INSTALL_DIR%\" >nul
    ) else (
        echo Warning: CUDA DLLs directory not found: !CUDA_DLLS!
    )
) else (
    echo Warning: CUDA_PATH not set. CUDA runtime DLLs will not be copied.
)

echo Installation complete.
goto DONE

:ERROR
set "EXITCODE=%errorlevel%"
popd
exit /B %EXITCODE%

:DONE
popd
exit /B 0
