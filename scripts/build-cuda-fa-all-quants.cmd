@echo off
setlocal

set "ACTION=%~1"
if not defined ACTION set "ACTION=all"

set /a DEFAULT_JOBS=%NUMBER_OF_PROCESSORS%
if %DEFAULT_JOBS% GTR 8 set /a DEFAULT_JOBS=%NUMBER_OF_PROCESSORS%/2
if %DEFAULT_JOBS% LSS 1 set "DEFAULT_JOBS=1"

set "JOBS=%~2"
if not defined JOBS set "JOBS=%DEFAULT_JOBS%"

set "CONFIG=%~3"
if not defined CONFIG set "CONFIG=Release"

set "BUILD_DIR=%~4"
if not defined BUILD_DIR set "BUILD_DIR=build-cuda-fa-all-quants"

pushd "%~dp0.."
if errorlevel 1 goto ERROR

if /I "%ACTION%"=="all" goto ALL
if /I "%ACTION%"=="configure" goto CONFIGURE
if /I "%ACTION%"=="build" goto BUILD
if /I "%ACTION%"=="clean" goto CLEAN
if /I "%ACTION%"=="help" goto HELP

echo Unknown action: %ACTION%
goto HELP

:ALL
call :CONFIGURE_STEP
if errorlevel 1 goto ERROR
call :BUILD_STEP
if errorlevel 1 goto ERROR
goto DONE

:CONFIGURE
call :CONFIGURE_STEP
if errorlevel 1 goto ERROR
goto DONE

:BUILD
call :BUILD_STEP
if errorlevel 1 goto ERROR
goto DONE

:CLEAN
if exist "%BUILD_DIR%" (
    rmdir /S /Q "%BUILD_DIR%"
    if errorlevel 1 goto ERROR
)
goto DONE

:CONFIGURE_STEP
echo Configuring %BUILD_DIR% with CUDA and all FA quant kernels enabled...
cmake -S . -B "%BUILD_DIR%" ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DGGML_CUDA_NCCL=OFF ^
  -DLLAMA_OPENSSL=OFF ^
  -DGGML_CCACHE=OFF ^
  -Wno-dev
exit /B %errorlevel%

:BUILD_STEP
echo Building %BUILD_DIR% [%CONFIG%] with %JOBS% parallel jobs...
cmake --build "%BUILD_DIR%" --config "%CONFIG%" --parallel %JOBS%
exit /B %errorlevel%

:HELP
echo Usage:
echo   scripts\build-cuda-fa-all-quants.cmd [all^|configure^|build^|clean^|help] [jobs] [config] [build_dir]
echo.
echo Examples:
echo   scripts\build-cuda-fa-all-quants.cmd
echo   scripts\build-cuda-fa-all-quants.cmd configure
echo   scripts\build-cuda-fa-all-quants.cmd build 16 Release
echo   scripts\build-cuda-fa-all-quants.cmd all 16 Release build-cuda-fa-all-quants
echo.
echo Defaults:
echo   action   = all
echo   jobs     = %DEFAULT_JOBS% ^(roughly physical cores; override if you want^)
echo   config   = Release
echo   build_dir= build-cuda-fa-all-quants
goto DONE

:ERROR
set "EXITCODE=%errorlevel%"
popd
exit /B %EXITCODE%

:DONE
popd
exit /B 0