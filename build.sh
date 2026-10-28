#!/usr/bin/env bash
#
# Compila PearlHost en Linux y deja ./out listo para ejecutar.
#
#   ./build.sh                          # detecta la GPU y compila su perfil
#   ./build.sh                          # PEARL_GEMM_ARCH=ampere ./build.sh
#   PEARL_GEMM_ARCH=all ./build.sh      # paquete completo (RTX/GTX)
#
# Estructura de salida (idéntica a la del build.ps1 de Windows):
#   out/pearl-host                     host Rust
#   out/libpearl_mining_capi.so        BLAKE3 / merkle
#   out/libcudart.so.12                runtime CUDA (si hay toolkit)
#   out/libs/<perfil>/libpearl_gemm_capi.so
#
# Requisitos: cargo, make, python3, git, CUDA 12.x (nvcc). En Docker usa
# la imagen nvidia/cuda:12.8-devel-ubuntu24.04.
#
# Nota: la ruta Windows es la validada a fondo; esta ruta Linux es
# experimental (misma lógica, menos horas de vuelo).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${PEARL_GEMM_ARCH:-auto}"
OUT="${OUT:-$ROOT/out}"
CONFIG="${CONFIG:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

ALL_ARCHS=(pascal turing volta ampere ada blackwell)

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '  - %s\n' "$*"; }

# nvcc >= 12 en PATH o en /usr/local/cuda*.
pick_nvcc() {
  local n v
  for n in "${NVCC:-}" $(command -v nvcc 2>/dev/null) /usr/local/cuda*/bin/nvcc /opt/cuda*/bin/nvcc; do
    [ -n "$n" ] && [ -x "$n" ] || continue
    # `|| true`: nvcc roto no debe abortar el script (set -e).
    v="$("$n" --version 2>/dev/null | grep -oE 'release [0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
    [ "${v:-0}" -ge 12 ] && { echo "$n"; return; }
  done
}

detect_arch() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
  case "$cap" in
    6.1)         echo pascal ;;
    7.0)         echo volta ;;
    7.5)         echo turing ;;
    8.0|8.6|8.7) echo ampere ;;
    8.9)         echo ada ;;
    9.0)         echo h100 ;;
    10.*)        echo b200 ;;
    12.*)        echo blackwell ;;
    *)           return 0 ;;
  esac
}

# ── Preflight ────────────────────────────────────────────────────────────────
say 'Comprobando herramientas'
miss=()
command -v cargo   >/dev/null 2>&1 || miss+=("cargo (Rust) -> https://rustup.rs")
command -v make    >/dev/null 2>&1 || miss+=("make -> apt install build-essential")
command -v python3 >/dev/null 2>&1 || miss+=("python3 -> apt install python3")
NVCC_BIN="$(pick_nvcc)"
[ -n "$NVCC_BIN" ] || miss+=("nvcc (CUDA >= 12) -> https://developer.nvidia.com/cuda-downloads")
if [ "${#miss[@]}" -gt 0 ]; then
  printf '\n\033[1;31mFaltan requisitos:\033[0m\n' >&2
  printf '  - %s\n' "${miss[@]}" >&2
  exit 1
fi
step "nvcc : $NVCC_BIN"
step "cargo: $(command -v cargo)"

# ── Perfiles ─────────────────────────────────────────────────────────────────
if [ "$ARCH" = "all" ]; then
  archs=("${ALL_ARCHS[@]}")
  say "Compilando todos los perfiles: ${archs[*]}"
elif [ "$ARCH" = "auto" ]; then
  detected="$(detect_arch)"
  if [ -n "$detected" ]; then
    archs=("$detected")
    say "GPU detectada -> perfil '$detected'"
  else
    archs=(ampere)
    printf '  No pude detectar la GPU; compilo ampere (RTX 30xx / A-series).\n' >&2
  fi
else
  archs=("$ARCH")
  say "Perfil forzado: $ARCH"
fi

nvcc_major="$("$NVCC_BIN" --version 2>/dev/null | grep -oE 'release [0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
for a in "${archs[@]}"; do
  case "$a" in
    pascal|volta)
      if [ "${nvcc_major:-0}" -ge 13 ]; then
        die "CUDA $nvcc_major ya no compila $a (sm_61/sm_70). Usa CUDA 12.8."
      fi ;;
  esac
done

# ── CUTLASS ──────────────────────────────────────────────────────────────────
if [ ! -e "$ROOT/native/pearl-gemm/third_party/cutlass/include/cutlass/cutlass.h" ]; then
  say 'Descargando CUTLASS (submódulo)'
  git -C "$ROOT" submodule update --init --depth 1 native/pearl-gemm/third_party/cutlass \
    || die 'no se pudo descargar CUTLASS (clona con --recurse-submodules)'
fi

# ── 1. BLAKE3 / merkle ───────────────────────────────────────────────────────
say 'Compilando libpearl_mining_capi.so (Rust)'
cargo build --release --manifest-path "$ROOT/native/Cargo.toml" -q
MINING_SO="$ROOT/native/target/release/libpearl_mining_capi.so"
[ -f "$MINING_SO" ] || die "no se generó $MINING_SO"

# ── 2. Host ──────────────────────────────────────────────────────────────────
say 'Compilando pearl-host (Rust)'
cargo build --release --manifest-path "$ROOT/host/Cargo.toml" -q
HOST_BIN="$ROOT/host/target/release/pearl-host"
[ -x "$HOST_BIN" ] || die "no se generó $HOST_BIN"

# ── 3. Motor CUDA por perfil ─────────────────────────────────────────────────
export PATH="$(dirname "$NVCC_BIN"):$PATH"
for a in "${archs[@]}"; do
  say "Compilando motor CUDA (perfil $a)"
  gemm_build="$ROOT/native/pearl-gemm/csrc/capi/build/$a"
  make_log="$(mktemp)"
  # El Makefile decide frescura por mtime y no ve las flags -D; un directorio
  # viejo puede enlazar objetos de otra arquitectura. Lo limpiamos siempre.
  rm -rf "$gemm_build"
  if ! make -C "$ROOT/native/pearl-gemm/csrc/capi" -j"$JOBS" \
       BUILD="$gemm_build" PEARL_GEMM_ARCH="$a" NVCC="$NVCC_BIN" >"$make_log" 2>&1; then
    tail -n 40 "$make_log" >&2
    rm -f "$make_log"
    die "make falló compilando $a"
  fi
  rm -f "$make_log"
  [ -f "$gemm_build/libpearl_gemm_capi.so" ] || die "no se generó la DLL de $a"
  step "$a -> $(du -h "$gemm_build/libpearl_gemm_capi.so" | cut -f1)"
done

# ── 4. Staging ───────────────────────────────────────────────────────────────
say "Preparando $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

cp "$HOST_BIN" "$OUT/pearl-host"
cp "$MINING_SO" "$OUT/libpearl_mining_capi.so"
chmod +x "$OUT/pearl-host"

# libcudart: el loader la busca junto al ejecutable o en el toolkit.
cudart="$(ls -1 /usr/local/cuda*/lib64/libcudart.so.12 2>/dev/null | head -1 || true)"
if [ -n "$cudart" ]; then
  cp -L "$cudart" "$OUT/libcudart.so.12"
else
  printf '  aviso: no encontré libcudart.so.12 (se buscará en /usr/local/cuda)\n' >&2
fi

for a in "${archs[@]}"; do
  mkdir -p "$OUT/libs/$a"
  cp "$ROOT/native/pearl-gemm/csrc/capi/build/$a/libpearl_gemm_capi.so" "$OUT/libs/$a/"
done

printf '\n\033[1;32mListo.\033[0m Contenido de out:\n'
find "$OUT" -type f | sed "s|^$OUT/|   |"

printf '\nEjecuta:\n'
printf '   %s --wallet prl1... --worker rig1\n' "$OUT/pearl-host"
printf '   %s --help\n' "$OUT/pearl-host"
