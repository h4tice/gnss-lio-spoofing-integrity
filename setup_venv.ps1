# Proje klasöründe .venv sanal ortamını oluşturur ve kütüphaneleri yalnızca bu ortama kurar.
# Kullanım (PowerShell):  .\setup_venv.ps1
# Betik çalıştırma engelliyse önce:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/2] .venv oluşturuluyor (Python 3.11)..."
    py -3.11 -m venv .venv
}
Write-Host "[2/2] Kütüphaneler .venv içine kuruluyor..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Kurulum tamam. Ortamı etkinleştirmek için:  .\.venv\Scripts\Activate.ps1"
Write-Host "Hızlı deneme:  python -m gnss_lio_sim all --quick"
