#!/usr/bin/env bash
#
# Traduce variables de entorno de Docker a argumentos de pearl-host.
#
#   PEARL_WALLET     wallet de pago (obligatoria)
#   PEARL_WORKER     nombre de worker           [docker]
#   PEARL_POOL_HOST  host del pool              [br.pearl.herominers.com]
#   PEARL_POOL_PORT  puerto del pool            [1200]
#   PEARL_LOWMEM=1   geometría reducida (3-4 GB de VRAM)
#   PEARL_GPU        índice de GPU              [0]
#   PEARL_SLOTS      nonces por batch           [16]
#   PEARL_EXTRA      argumentos extra (texto plano, se separan por espacios)
#
# Cualquier argumento pasado al contenedor se añade tal cual:
#   docker run ... pearlhost --selfcheck
set -euo pipefail

args=()
[ -n "${PEARL_WALLET:-}"    ] && args+=(--wallet "$PEARL_WALLET")
[ -n "${PEARL_WORKER:-}"    ] && args+=(--worker "$PEARL_WORKER")
[ -n "${PEARL_POOL_HOST:-}" ] && args+=(--host "$PEARL_POOL_HOST")
[ -n "${PEARL_POOL_PORT:-}" ] && args+=(--port "$PEARL_POOL_PORT")
[ -n "${PEARL_GPU:-}"       ] && args+=(--gpu "$PEARL_GPU")
[ -n "${PEARL_SLOTS:-}"     ] && args+=(--slots "$PEARL_SLOTS")
[ "${PEARL_LOWMEM:-0}" = "1" ] && args+=(--lowmem)
# shellcheck disable=SC2206
[ -n "${PEARL_EXTRA:-}"     ] && args+=(${PEARL_EXTRA})

exec /app/pearl-host "${args[@]}" "$@"
