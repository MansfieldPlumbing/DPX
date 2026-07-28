$vsPath = "S:\bin\vs"
$sdkPath = "S:\bin\Windows Kits\10"
$msvcDir = Get-ChildItem -Path "$vsPath\VC\Tools\MSVC" -Directory | Select-Object -First 1
$msvcPath = $msvcDir.FullName
$sdkVerDir = Get-ChildItem -Path "$sdkPath\Lib" -Directory | Where-Object { $_.Name -like "10.*" } | Sort-Object Name -Descending | Select-Object -First 1
$sdkVer = $sdkVerDir.Name

$env:INCLUDE = "$msvcPath\include;$sdkPath\Include\$sdkVer\ucrt;$sdkPath\Include\$sdkVer\um;$sdkPath\Include\$sdkVer\shared;src"
$env:LIB = "$msvcPath\lib\x64;$sdkPath\Lib\$sdkVer\ucrt\x64;$sdkPath\Lib\$sdkVer\um\x64"
$env:PATH = "$msvcPath\bin\Hostx64\x64;" + $env:PATH

cl /nologo /EHsc /Isrc check_mask.cpp src\sqlite3.c /Fe:check_mask.exe
if ($LASTEXITCODE -eq 0) {
    .\check_mask.exe
}
Remove-Item check_mask.* -Force -ErrorAction SilentlyContinue
