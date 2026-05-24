@echo off
setlocal

REM === Guard: ensure we're on the alefranz branch ===
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD') do set "CURRENT_BRANCH=%%i"
if /I not "%CURRENT_BRANCH%"=="alefranz" (
    echo Error: This script must be run from the ^"alefranz^" branch.
    echo Current branch: %CURRENT_BRANCH%
    exit /B 1
)

echo === Fetching upstream master ===
git fetch upstream master:master
if errorlevel 1 (
    echo Error: Failed to fetch upstream master.
    exit /B 1
)

echo === Pushing local master to origin ===
git push origin master
if errorlevel 1 (
    echo Error: Failed to push master to origin.
    exit /B 1
)

echo === Interactive rebase onto master ===
git rebase master
if errorlevel 1 (
    echo Error: Rebase failed or was aborted.
    exit /B 1
)

echo === Force pushing alefranz to origin ===
git push origin HEAD --force-with-lease
if errorlevel 1 (
    echo Error: Failed to force push to origin.
    exit /B 1
)

echo Done.
exit /B 0
