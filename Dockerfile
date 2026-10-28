# syntax=docker/dockerfile:1.7
#
# Imagen de PearlHost — minero de Pearl (PRL) para NVIDIA.
#
# La imagen trae el host Rust y el motor CUDA compilado para varios perfiles
# (pascal, turing, volta, ampere, ada, blackwell). El host detecta la GPU en
# tiempo de ejecución y elige libs/<perfil>/ automáticamente.
#
# Uso:
#   docker build -t pearlhost .
#   docker run --gpus all -e PEARL_WALLET=prl1... pearlhost
#   docker run --gpus all -e PEARL_WALLET=prl1... -e PEARL_WORKER=rig1 pearlhost
#
# Nota: esta ruta (Linux/Docker) es experimental. La ruta Windows
# (build.ps1) es la validada a fondo.

ARG CUDA_VERSION=12.8.1
ARG CUDA_UBUNTU=ubuntu24.04
ARG RUST_VERSION=1.83
# CUTLASS v4.8.1 (mismo commit que el submódulo del repo).
ARG CUTLASS_REF=91843b71d90737e0a9b86f7d9df213cff5a3526f
ARG GEMM_ARCHS=all

# ── Build ────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-${CUDA_UBUNTU} AS builder

ARG RUST_VERSION
ARG CUTLASS_REF
ARG GEMM_ARCHS

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git make python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust:${RUST_VERSION} /usr/local/cargo  /usr/local/cargo
COPY --from=rust:${RUST_VERSION} /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}" \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo

WORKDIR /src
COPY . .
RUN rm -rf native/pearl-gemm/third_party/cutlass out host/target native/target

# CUTLASS en el commit exacto que espera el motor.
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git \
      native/pearl-gemm/third_party/cutlass \
    && cd native/pearl-gemm/third_party/cutlass \
    && git fetch --depth 1 origin "${CUTLASS_REF}" \
    && git checkout --detach "${CUTLASS_REF}"

RUN chmod +x build.sh docker-entrypoint.sh \
    && PEARL_GEMM_ARCH="${GEMM_ARCHS}" ./build.sh

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${CUDA_UBUNTU} AS final

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tini bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/out/ /app/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/pearl-host /app/docker-entrypoint.sh

ENV LD_LIBRARY_PATH=/app \
    PEARL_POOL_HOST=br.pearl.herominers.com \
    PEARL_POOL_PORT=1200 \
    PEARL_WORKER=docker

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD /app/pearl-host --version > /dev/null 2>&1 || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]
CMD []
