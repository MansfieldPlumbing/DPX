#Requires -Version 7.0
$ErrorActionPreference = 'Stop'

Write-Host ">>> Configuring Native Engine Environments..." -ForegroundColor Cyan

# 1. Grab SQLite amalgamation safely if it's missing (Prevents CMake linker crashes on Windows)
if (-not (Test-Path "sqlite3.c")) {
    Write-Host ">>> SQLite amalgamation missing. Downloading zero-config dependencies..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://sqlite.org/2024/sqlite-amalgamation-3450200.zip" -OutFile "sqlite.zip"
            Expand-Archive -Path "sqlite.zip" -DestinationPath "sqlite_tmp" -Force
                
    Copy-Item -Path "sqlite_tmp\*\sqlite3.c" -Destination "." -Force
        Copy-Item -Path "sqlite_tmp\*\sqlite3.h" -Destination "." -Force
            
    Remove-Item "sqlite.zip" -Force
        Remove-Item "sqlite_tmp" -Recurse -Force
            Write-Host ">>> Dropped sqlite3.c and sqlite3.h natively." -ForegroundColor Green
            }
            
# 2. Check for CMake Environment Target Space
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "FATAL: CMake not found in PATH. Ensure Desktop C++ toolchains are mounted." -ForegroundColor Red
        exit 1
        }
        
# 3. Secure output build zone
if (-not (Test-Path build)) { New-Item -ItemType Directory -Name build | Out-Null }
Set-Location build

try {
    Write-Host "`n>>> Evaluating CMa8e Linkages..."
        cmake .. -G "Visual Studio 17 2026" -A x64 
            if ($LASTEXITCODE -ne 0) { throw "CMake evaluation dropped payload targets." }
            
    Write-Host "`n>>> Saturated Core Parallel Compiler Pipeline Executing..."
        cmake --build . --config Release -- /m
            if ($LASTEXITCODE -ne 0) { throw "Compiler assembly crash detected." }
            
    $dllPath = "Release/dpx_engine.dll"
        if (Test-Path $dllPath) {
                Write-Host "`n>>> DLL BINARY SECURED. [.\build\Release\dpx_engine.dll]" -ForegroundColor Green
                    } else {
                            throw "Compile complete but dynamic link output inaccessible."
                                }
                                } catch {
                                    Write-Host "`n>>> COMPILE BURN. See exception cascade above." -ForegroundColor Red
                                        exit 1
                                        } finally {
                                            Set-Location ..
                                            }
                                            