#Requires -Version 7.0
$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  DPX COMPILER - DUAL ARTIFACT (DLL + EXE)" -ForegroundColor Cyan
Write-Host "================================================================================"

# Setup environment variables manually for portable / custom VS and Windows SDK installation
Write-Host "  [+] Configuring portable/custom MSVC and Windows SDK environment..." -ForegroundColor Yellow

# Setup environment variables manually for portable / custom VS and Windows SDK installation
Write-Host "  [+] Configuring portable/custom MSVC and Windows SDK environment..." -ForegroundColor Yellow

$vsPath = "S:\bin\vs"
$sdkPath = "S:\bin\Windows Kits\10"

if ((Test-Path $vsPath) -and (Test-Path $sdkPath)) {
    # 1. Resolve MSVC version-specific path
    $msvcDir = Get-ChildItem -Path "$vsPath\VC\Tools\MSVC" -Directory | Select-Object -First 1
    if (-not $msvcDir) { throw "Could not find MSVC directory in S:\Dev\bin\vs\VC\Tools\MSVC" }
    $msvcPath = $msvcDir.FullName
    Write-Host "      Found MSVC: $($msvcDir.Name)" -ForegroundColor DarkGray

    # 2. Resolve Windows SDK version
    $sdkVerDir = Get-ChildItem -Path "$sdkPath\Lib" -Directory | Where-Object { $_.Name -like "10.*" } | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $sdkVerDir) { throw "Could not find a valid Windows 10/11 SDK version under S:\Dev\bin\Windows Kits\10\Lib" }
    $sdkVer = $sdkVerDir.Name
    Write-Host "      Found Windows SDK: $sdkVer" -ForegroundColor DarkGray

    # 3. Configure INCLUDE paths
    $includePaths = @(
        "$msvcPath\include",
        "$sdkPath\Include\$sdkVer\ucrt",
        "$sdkPath\Include\$sdkVer\um",
        "$sdkPath\Include\$sdkVer\shared",
        "$sdkPath\Include\$sdkVer\winrt",
        "$sdkPath\Include\$sdkVer\cppwinrt"
    )
    $env:INCLUDE = ($includePaths -join ";") + ";" + $env:INCLUDE

    # 4. Configure LIB paths
    $libPaths = @(
        "$msvcPath\lib\x64",
        "$sdkPath\Lib\$sdkVer\ucrt\x64",
        "$sdkPath\Lib\$sdkVer\um\x64"
    )
    $env:LIB = ($libPaths -join ";") + ";" + $env:LIB

    # 5. Prepend PATH with the compiler executables and SDK tools
    $compilerBin = "$msvcPath\bin\Hostx64\x64"
    $sdkBin = "$sdkPath\bin\$sdkVer\x64"
    $env:PATH = "$compilerBin;$sdkBin;" + $env:PATH
    
    Write-Host "  [+] Environment configured successfully!" -ForegroundColor Green
} else {
    Write-Error "S:\Dev\bin\vs or S:\Dev\bin\Windows Kits\10 is missing. Check your installation paths."
    exit 1
}

$coreSources = Get-ChildItem -Path "." -Filter "*.cpp" | Where-Object { $_.Name -ne "main.cpp" -and $_.Name -ne "dpx_api_mount.cpp" -and $_.Name -ne "firstlight.cpp" -and $_.Name -ne "chain.cpp" -and $_.Name -ne "gemm.cpp" } | Select-Object -ExpandProperty FullName
$cSources = Get-ChildItem -Path "." -Filter "*.c" | Select-Object -ExpandProperty FullName

Write-Host "  [+] Compiling Core Object Files..." -ForegroundColor DarkGray
& cl.exe /c /EHsc /O2 /std:c++17 /I. $coreSources $cSources
if ($LASTEXITCODE -ne 0) { Write-Error "Core compilation failed."; exit 1 }

$objFiles = Get-ChildItem -Path "." -Filter "*.obj" | Where-Object { $_.Name -ne "main.obj" -and $_.Name -ne "dpx_api_mount.obj" } | Select-Object -ExpandProperty Name

Write-Host "  [+] Compiling & Linking dpx_engine.dll..." -ForegroundColor Cyan
& cl.exe /EHsc /O2 /std:c++17 /LD /I. dpx_api_mount.cpp $objFiles /link /out:dpx_engine.dll d3d12.lib dxgi.lib synchronization.lib
if ($LASTEXITCODE -ne 0) { Write-Error "DLL linkage failed."; exit 1 }

Write-Host "  [+] Compiling & Linking dpx.exe..." -ForegroundColor Cyan
& cl.exe /EHsc /O2 /std:c++17 /I. main.cpp $objFiles /link /out:dpx.exe d3d12.lib dxgi.lib synchronization.lib
if ($LASTEXITCODE -ne 0) { Write-Error "EXE linkage failed."; exit 1 }

Write-Host "`n  [OK] Successfully built dpx_engine.dll AND dpx.exe!" -ForegroundColor Green
