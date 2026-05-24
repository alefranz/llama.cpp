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

set "INSTALL_DIR=%~5"
if not defined INSTALL_DIR set "INSTALL_DIR=C:\Software\llama\current"

set "SCRIPT_DIR=%~dp0"

pushd "%~dp0.."
if errorlevel 1 goto ERROR

if /I "%ACTION%"=="all" goto ALL
if /I "%ACTION%"=="help" goto HELP

echo Unknown action: %ACTION%
goto HELP

:ALL
echo [1/4] Rebasing alefranz branch...
call "%SCRIPT_DIR%rebase-cuda-fa-all-quants.cmd"
if errorlevel 1 goto ERROR

echo [2/4] Configuring...
call "%SCRIPT_DIR%build-cuda-fa-all-quants.cmd" configure %JOBS% %CONFIG% %BUILD_DIR%
if errorlevel 1 goto ERROR

echo [3/4] Building...
call "%SCRIPT_DIR%build-cuda-fa-all-quants.cmd" build %JOBS% %CONFIG% %BUILD_DIR%
if errorlevel 1 goto ERROR

echo [4/4] Installing...
call "%SCRIPT_DIR%install-cuda-fa-all-quants.cmd" %BUILD_DIR% %CONFIG% %INSTALL_DIR%
if errorlevel 1 goto ERROR

echo Done.
goto DONE

:HELP
echo Usage:
echo   scripts\rebase-build-install.cmd [all^|help] [jobs] [config] [build_dir] [install_dir]
echo.
echo Examples:
echo   scripts\rebase-build-install.cmd
echo   scripts\rebase-build-install.cmd all 16 Release build-cuda-fa-all-quants C:\Software\llama\current
echo.
echo Steps:
echo   1. Rebase alefranz onto upstream master (rebase-cuda-fa-all-quants.cmd)
echo   2. Configure build (cmake)
echo   3. Build (build-cuda-fa-all-quants.cmd)
echo   4. Install (install-cuda-fa-all-quants.cmd)
echo.
echo Defaults:
echo   action      = all
echo   jobs        = %DEFAULT_JOBS%
echo   config      = Release
echo   build_dir   = build-cuda-fa-all-quants
echo   install_dir = C:\Software\llama\current
goto DONE

:ERROR
set "EXITCODE=%errorlevel%"
popd
exit /B %EXITCODE%

:DONE
popd
exit /B 0
