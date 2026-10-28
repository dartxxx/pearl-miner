<#
.SYNOPSIS
  Compila PearlHost (minero de Pearl / PRL) en Windows nativo.

.DESCRIPTION
  Genera una carpeta ./out lista para ejecutar:

    out/pearl-host.exe              host Rust (stratum v2, commitments, fee)
    out/pearl_mining_capi.dll       BLAKE3 keyed-merkle (Rust)
    out/cudart64_12.dll             runtime CUDA (copiado del toolkit)
    out/libs/<perfil>/pearl_gemm_capi.dll
                                    motor CUDA compilado por arquitectura

  El host detecta la compute capability real de la GPU y elige la DLL de
  libs/ automáticamente, así que puedes distribuir un solo paquete con
  varios perfiles dentro.

  Requisitos: Visual Studio con "Desktop development with C++" (cl.exe),
  CUDA Toolkit 12.x (nvcc), Rust (cargo) y python (codegen de kernels).
  CUTLASS se descarga como submódulo git si falta.

.PARAMETER Arch
  Perfil a compilar. 'auto' (por defecto) detecta la GPU instalada.

.PARAMETER All
  Compila todos los perfiles de consumo: pascal, turing, volta, ampere,
  ada y blackwell.

.EXAMPLE
  .\build.ps1                     # detecta tu GPU y compila su perfil
  .\build.ps1 -All                # paquete completo para cualquier RTX/GTX
  .\build.ps1 -Arch ampere        # fuerza RTX 30xx / A-series
  .\build.ps1 -Arch pascal -NoVerify
#>
[CmdletBinding()]
param(
  [ValidateSet('auto','h100','volta','turing','pascal','portable','ampere','ada','blackwell','b200')]
  [string]$Arch = $(if ($env:PEARL_GEMM_ARCH) { $env:PEARL_GEMM_ARCH } else { 'auto' }),

  [switch]$All,

  [ValidateSet('Release','Debug')]
  [string]$Config = 'Release',

  [string]$Out,

  # Carpeta bin del toolkit CUDA (por defecto: CUDA_PATH o la v12.x más nueva).
  [string]$CudaBin = $env:CUDA_BIN,

  # GPU usada para la autodetección cuando hay varias.
  [int]$Gpu = 0,

  # nvcc solo acepta MSVC de VS 2019-2022. 'auto' lo activa si hace falta.
  [ValidateSet('auto','on','off')]
  [string]$AllowUnsupportedCompiler = 'auto',

  # No ejecutar --list-profiles al terminar.
  [switch]$NoVerify
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root 'out' }

# Perfiles que se incluyen con -All (RTX/GTX de consumo).
$ALL_ARCHS = @('pascal','turing','volta','ampere','ada','blackwell')

function Say  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Die  ($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }
function Step ($m) { Write-Host "  - $m" -ForegroundColor DarkCyan }

# ── Visual Studio ────────────────────────────────────────────────────────────
function Find-VsInstall {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { return $null }
  $path = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath 2>$null | Select-Object -First 1
  if (-not $path) { return $null }
  [pscustomobject]@{
    Path      = $path
    VcVars    = Join-Path $path 'VC\Auxiliary\Build\vcvars64.bat'
    Installer = Split-Path $vswhere
  }
}

# Importa el entorno de vcvars64.bat (cl.exe, link.exe, include/lib paths).
function Import-VcVars ($vcvars) {
  cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([A-Za-z_][A-Za-z0-9_()]*)=(.*)$') {
      Set-Item -Path "Env:\$($matches[1])" -Value $matches[2]
    }
  }
  # CMake prefiere estas si están puestas y apuntan a otro compilador.
  Remove-Item Env:\CXX, Env:\CC -ErrorAction SilentlyContinue
}

# ── CUDA ─────────────────────────────────────────────────────────────────────
function Find-Cuda {
  param([string]$Explicit)
  $cands = @()
  if ($Explicit) { $cands += $Explicit }
  if ($env:CUDA_BIN) { $cands += $env:CUDA_BIN }
  if ($env:CUDA_PATH) { $cands += (Join-Path $env:CUDA_PATH 'bin') }

  $rootCuda = Join-Path ${env:ProgramFiles} 'NVIDIA GPU Computing Toolkit\CUDA'
  if (Test-Path $rootCuda) {
    $dirs = Get-ChildItem $rootCuda -Directory | Sort-Object {
      if ($_.Name -match '^v(\d+)\.(\d+)') { [version]"$($matches[1]).$($matches[2])" } else { [version]'0.0' }
    } -Descending
    foreach ($d in $dirs) { $cands += (Join-Path $d.FullName 'bin') }
  }

  foreach ($c in $cands) {
    if ($c -and (Test-Path (Join-Path $c 'nvcc.exe'))) {
      return [pscustomobject]@{ Bin = $c; Nvcc = (Join-Path $c 'nvcc.exe') }
    }
  }

  $gc = Get-Command nvcc -ErrorAction SilentlyContinue
  if ($gc) { return [pscustomobject]@{ Bin = (Split-Path $gc.Source); Nvcc = $gc.Source } }
  return $null
}

function Nvcc-Major ($nvcc) {
  $out = (& $nvcc --version 2>$null) -join "`n"
  if ($out -match 'release (\d+)') { return [int]$matches[1] }
  return 0
}

# ── Arquitectura de la GPU → perfil ──────────────────────────────────────────
function Detect-Arch {
  param([int]$Index = 0)
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $smi) { return '' }

  $cap = (& $smi --query-gpu=compute_cap --format=csv,noheader,nounits -i $Index 2>$null |
          Select-Object -First 1)
  if ($cap) { $cap = $cap.Trim() }

  switch -Regex ($cap) {
    '^6\.1$'        { return 'pascal'    }
    '^6\.'          { return 'pascal'    }   # 6.0/6.2 sin DP4A → el host avisará
    '^7\.0$'        { return 'volta'     }
    '^7\.5$'        { return 'turing'    }
    '^8\.(0|6|7)$'  { return 'ampere'    }
    '^8\.9$'        { return 'ada'       }
    '^9\.0$'        { return 'h100'      }
    '^10\.'         { return 'b200'      }
    '^12\.'         { return 'blackwell' }
  }

  # Driver viejo sin `compute_cap`: caemos al nombre comercial.
  $name = (& $smi --query-gpu=name --format=csv,noheader -i $Index 2>$null |
           Select-Object -First 1)
  if (-not $name) { return '' }
  switch -Regex ($name) {
    'H100|H200'                       { return 'h100'      }
    'B100|B200|GB200'                 { return 'b200'      }
    'RTX\s*50\d\d'                    { return 'blackwell' }
    'RTX\s*40\d\d|L40|L4|\bA?6000 Ada' { return 'ada'       }
    'RTX\s*30\d\d|A100|A40|A6000|A5000|A2000' { return 'ampere' }
    'RTX\s*20\d\d|GTX\s*16\d\d|T4|Quadro\s*T' { return 'turing' }
    'GTX\s*10\d\d|P40|P100|P4|P6'     { return 'pascal'    }
    'Titan\s*V|V100'                  { return 'volta'     }
  }
  return ''
}

# ── Herramientas ─────────────────────────────────────────────────────────────
function Resolve-Tool ($name, $vsRelPaths, $vsPath) {
  $c = Get-Command $name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  foreach ($rel in $vsRelPaths) {
    $p = Join-Path $vsPath $rel
    if (Test-Path $p) { return $p }
  }
  return $null
}

# Un CMakeCache.txt de otra ruta fuente/destino produce DLLs sin código para la
# GPU actual (cudaErrorNoKernelImageForDevice = -209). Lo detectamos y borramos.
function Clear-StaleCache([string]$buildDir, [string]$srcDir) {
  $cache = Join-Path $buildDir 'CMakeCache.txt'
  if (-not (Test-Path $cache)) { return $false }

  $norm = {
    param($p)
    if (-not $p) { return '' }
    try { ([IO.Path]::GetFullPath($p.Trim())).Replace('\', '/').TrimEnd('/').ToLowerInvariant() }
    catch { $p.ToLowerInvariant() }
  }

  $homeLine = Select-String -Path $cache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$' | Select-Object -First 1
  $dirLine  = Select-String -Path $cache -Pattern '^CMAKE_CACHEFILE_DIR:INTERNAL=(.*)$' | Select-Object -First 1

  $cachedHome = if ($homeLine) { & $norm $homeLine.Matches[0].Groups[1].Value } else { '' }
  $cachedDir  = if ($dirLine)  { & $norm $dirLine.Matches[0].Groups[1].Value  } else { '' }
  $wantHome = & $norm $srcDir
  $wantDir  = & $norm $buildDir

  if ($cachedHome -ne $wantHome -or $cachedDir -ne $wantDir) {
    Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
    return $true
  }
  return $false
}

# ── Preflight ────────────────────────────────────────────────────────────────
Say 'Comprobando herramientas'
$miss = @()

$vs = Find-VsInstall
if (-not $vs) {
  $miss += 'Visual Studio con "Desktop development with C++" -> https://visualstudio.microsoft.com'
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
  $miss += 'cargo (Rust) -> https://rustup.rs'
}
if (-not (Get-Command python -ErrorAction SilentlyContinue) -and
    -not (Get-Command py -ErrorAction SilentlyContinue)) {
  $miss += 'python (codegen de kernels CUDA) -> https://python.org'
}

$cuda = Find-Cuda $CudaBin
if (-not $cuda) {
  $miss += 'nvcc (CUDA Toolkit 12.x) -> https://developer.nvidia.com/cuda-downloads'
}

$cmake = $null; $ninja = $null
if ($vs) {
  $cmake = Resolve-Tool 'cmake' @('Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe') $vs.Path
  $ninja = Resolve-Tool 'ninja' @('Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe') $vs.Path
  if (-not $cmake) { $miss += 'cmake -> instala el componente "C++ CMake tools" de VS, o https://cmake.org' }
  if (-not $ninja) { $miss += 'ninja -> instala el componente "C++ CMake tools" de VS' }
}

if ($miss.Count -gt 0) {
  Write-Host "`nFaltan requisitos:" -ForegroundColor Red
  $miss | ForEach-Object { Write-Host "  - $_" }
  Die 'Instálalos y vuelve a ejecutar .\build.ps1'
}

Import-VcVars $vs.VcVars
$env:PATH = "$($cuda.Bin);$(Split-Path $ninja);$env:PATH"

$nvccMajor = Nvcc-Major $cuda.Nvcc
Step "Visual Studio : $($vs.Path)"
Step "cmake         : $cmake"
Step "ninja         : $ninja"
Step "nvcc          : $($cuda.Nvcc) (CUDA $nvccMajor)"
if ($nvccMajor -lt 12) { Die "se necesita CUDA 12 o superior (nvcc usa -std=c++20)" }

# ── Arquitecturas a compilar ─────────────────────────────────────────────────
if ($All) {
  $archs = $ALL_ARCHS
  Say "Compilando todos los perfiles: $($archs -join ', ')"
} elseif ($Arch -eq 'auto') {
  $detected = Detect-Arch $Gpu
  if ($detected) {
    $archs = @($detected)
    Say "GPU detectada -> perfil '$detected'"
  } else {
    $archs = @('ampere')
    Write-Host "  No pude detectar la GPU; compilo 'ampere' (RTX 30xx / A-series)." -ForegroundColor Yellow
    Write-Host "  Para otro modelo: -Arch pascal|turing|volta|ada|blackwell  (o -All)." -ForegroundColor Yellow
  }
} else {
  $archs = @($Arch)
  Say "Perfil forzado: $Arch"
}

# CUDA 13 eliminó la compilación offline de sm_61 (Pascal) y sm_70 (Volta).
$legacy = $archs | Where-Object { $_ -in @('pascal', 'volta') }
if ($legacy -and $nvccMajor -ge 13) {
  Die ("CUDA $nvccMajor ya no compila $($legacy -join '/') (sm_61/sm_70).`n" +
       "  Usa el toolkit CUDA 12.8 para esos perfiles, o compila solo ampere/ada/blackwell.")
}

# ── CUTLASS (submódulo) ──────────────────────────────────────────────────────
$cutlassHdr = Join-Path $root 'native\pearl-gemm\third_party\cutlass\include\cutlass\cutlass.h'
if (-not (Test-Path $cutlassHdr)) {
  Say 'Descargando CUTLASS (submódulo)'
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Die 'falta CUTLASS y no hay git. Clona con --recurse-submodules.'
  }
  git -C $root submodule update --init --depth 1 native/pearl-gemm/third_party/cutlass
  if ($LASTEXITCODE -ne 0) { Die 'no se pudo descargar CUTLASS' }
}

# ── 1. BLAKE3 / merkle (Rust) ────────────────────────────────────────────────
Say 'Compilando pearl_mining_capi.dll (Rust)'
cargo build --release --manifest-path (Join-Path $root 'native\Cargo.toml') -q
if ($LASTEXITCODE -ne 0) { Die 'cargo build (native) falló' }
$miningDll = Join-Path $root 'native\target\release\pearl_mining_capi.dll'
if (-not (Test-Path $miningDll)) { Die "no se generó $miningDll" }

# ── 2. Host (Rust) ───────────────────────────────────────────────────────────
Say 'Compilando pearl-host.exe (Rust)'
# CRT estático: el .exe no depende del VC++ Redistributable.
$oldFlags = $env:RUSTFLAGS
$env:RUSTFLAGS = (@($oldFlags, '-C target-feature=+crt-static') | Where-Object { $_ }) -join ' '
cargo build --release --manifest-path (Join-Path $root 'host\Cargo.toml') -q
$hostRc = $LASTEXITCODE
$env:RUSTFLAGS = $oldFlags
if ($hostRc -ne 0) { Die 'cargo build (host) falló' }
$hostExe = Join-Path $root 'host\target\release\pearl-host.exe'
if (-not (Test-Path $hostExe)) { Die "no se generó $hostExe" }

# ── 3. Motor CUDA, una DLL por perfil ────────────────────────────────────────
$cudaFlags = @()
$useAllow = $AllowUnsupportedCompiler
if ($useAllow -eq 'auto') {
  # VCToolsVersion = "14.44.35207" → MSVC 19.44 (sin invocar cl.exe, que
  # escribe el banner por stderr y rompe el pipeline).
  $toolset = "$env:VCToolsVersion"
  $msvc = if ($toolset -match '^14\.(\d+)') { [version]"19.$($matches[1])" } else { [version]'19.0' }
  $useAllow = if ($msvc -gt [version]'19.44') { 'on' } else { 'off' }
}
if ($useAllow -eq 'on') {
  Step "MSVC fuera del rango soportado por nvcc -> -allow-unsupported-compiler"
  $cudaFlags += '-allow-unsupported-compiler'
}

$gemmSrc = Join-Path $root 'native\pearl-gemm\csrc\capi'
$gemmDlls = @{}
foreach ($a in $archs) {
  Say "Compilando motor CUDA (perfil $a)"
  $gemmBuild = Join-Path $gemmSrc "build-win\$a"
  if (Clear-StaleCache $gemmBuild $gemmSrc) {
    Step "CMakeCache obsoleto en build-win\$a -> limpiado"
  }

  $cfgArgs = @('-S', $gemmSrc, '-B', $gemmBuild, '-G', 'Ninja',
               "-DPEARL_GEMM_ARCH=$a", "-DCMAKE_BUILD_TYPE=$Config")
  if ($cudaFlags.Count -gt 0) { $cfgArgs += "-DCMAKE_CUDA_FLAGS=$($cudaFlags -join ' ')" }

  & $cmake @cfgArgs
  if ($LASTEXITCODE -ne 0) { Die "CMake configure falló ($a)" }
  & $cmake --build $gemmBuild
  if ($LASTEXITCODE -ne 0) { Die "CMake build falló ($a)" }

  $dll = Join-Path $gemmBuild 'pearl_gemm_capi.dll'
  if (-not (Test-Path $dll)) { Die "no se generó $dll" }
  $gemmDlls[$a] = $dll
  Step ("$a -> {0:N2} MiB" -f ((Get-Item $dll).Length / 1MB))
}

# ── 4. Staging en ./out ──────────────────────────────────────────────────────
Say "Preparando $Out"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
New-Item -ItemType Directory -Path $Out | Out-Null

Copy-Item $hostExe $Out -Force
Copy-Item $miningDll $Out -Force

$cudart = Join-Path $cuda.Bin 'cudart64_12.dll'
if (Test-Path $cudart) {
  Copy-Item $cudart $Out -Force
} else {
  Write-Host "  aviso: no encontré cudart64_12.dll en $($cuda.Bin)" -ForegroundColor Yellow
  Write-Host "        el .exe lo buscará en el PATH (driver/toolkit)" -ForegroundColor Yellow
}

foreach ($a in $archs) {
  $dst = Join-Path $Out "libs\$a"
  New-Item -ItemType Directory -Path $dst -Force | Out-Null
  Copy-Item $gemmDlls[$a] $dst -Force
}

Write-Host "`nListo. Contenido de out:" -ForegroundColor Green
Get-ChildItem $Out -Recurse -File | ForEach-Object {
  Write-Host ("   {0}  ({1:N2} MiB)" -f $_.FullName.Substring($Out.Length + 1), ($_.Length / 1MB))
}

if (-not $NoVerify) {
  Write-Host "`nPerfiles en el paquete:" -ForegroundColor Green
  & (Join-Path $Out 'pearl-host.exe') --list-profiles
}

Write-Host "`nEjecuta:" -ForegroundColor Green
Write-Host "   .\out\pearl-host.exe --wallet prl1... --worker rig1"
Write-Host "   .\out\pearl-host.exe --help"
