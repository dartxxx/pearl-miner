# PearlHost — minero de Pearl (PRL) para NVIDIA

**Minero de Pearl (PRL) para GPUs NVIDIA. No necesitás saber programar — solo descargá y ejecutá.**

---

## ⚡ Instalación rápida (usuarios)

### 1. Descargá el minero

👉 **[Descargar última versión](https://github.com/dartxxx/pearl-miner/releases/latest)**

Extraé el `.zip` en una carpeta (ej: `C:\PearlMiner`).

### 2. Conseguí una wallet de Pearl (PRL)

- Creá una wallet en [Pearl Wallet](https://pearlresearch.ai) o en un exchange que soporte PRL (BigOne, HTX).
- Tu dirección empieza con `prl1...`

### 3. Ejecutá el minero

**Opción A — Doble clic (más fácil):**
1. Abrí la carpeta donde extrajiste el minero.
2. Editá `miner_live.bat` con Bloc de notas.
3. Reemplazá la wallet de ejemplo por la tuya.
4. Guardá y hacé doble clic en `miner_live.bat`.

**Opción B — Línea de comandos:**
```powershell
.\out\pearl-host.exe --wallet prl1TU_DIRECCION --worker mi_pc
```

### 4. Verificá que funciona

Deberías ver algo así:
```
  ✓ stratum: autorizado
  job    : 00000003_2097152  height=134722  cert_version=3
  ▶ minando (job listo en 0.8 s)
  stats 3.90 it/s · 32.8 TH/s · ok=1 rech=0
  gpu  : 110.2, 62, 45, 1920 (W,°C,%,MHz)
```

Si ves `ok=1` o más, ¡estás minando! 🎉

### Requisitos mínimos

- Windows 10/11
- GPU NVIDIA (GTX 1060 o superior)
- Driver NVIDIA actualizado
- Conexión a internet

### Preguntas frecuentes

**¿Cuánto gano?** Depende de tu GPU y del precio de PRL. Usá [WhatToMine](https://whattomine.com) para estimar.

**¿Gasta mucha electricidad?** El minero muestra consumo en watts en tiempo real. Calculá: watts × horas × precio/kWh.

**¿Puedo minar con la PC mientras uso otras apps?** Sí, pero el hashrate baja. Para mejor rendimiento, cerrá apps que usen la GPU.

**¿El fee del 1% es obligatorio?** No. Usá `--no-fee` para desactivarlo. Pero si te sirve el minero, dejalo activado — mantiene el proyecto vivo.

---

## 👨‍💻 Para desarrolladores

Minero de **Pearl (PRL)** para GPUs NVIDIA, escrito desde cero sobre el motor
CUDA de código abierto de [Akoya Pool](https://akoyapool.com) (con el parche
`cert-v3` que exige la red actual).

- Host en **Rust** (stratum v2, commitments BLAKE3, shares, estadísticas).
- Motor CUDA con **una DLL por arquitectura** y selección automática según la
  compute capability real de la tarjeta.
- Funciona en Windows (ruta validada) y Linux/Docker (experimental).
- **Fee de desarrollo: 1 %** del tiempo de minado (ver [Fee](#fee-de-desarrollo)).

---

## GPUs soportadas

El host consulta la GPU con `cudart` (compute capability, VRAM libre) y carga
`libs/<perfil>/pearl_gemm_capi.dll` automáticamente.

| Arquitectura | GPUs típicas | SM | Perfil | Estado |
|---|---|---|---|---|
| Pascal | GTX 1050/1060/1070/1080, P40 | 6.1 | `pascal` | Experimental (solo DP4A, lento) |
| Volta | Titan V, V100 | 7.0 | `volta` | Experimental (DP4A) |
| Turing | GTX 1650/1660, RTX 2060/2070/2080 | 7.5 | `turing` | Funcional (DP4A, sin tensor cores) |
| Ampere | RTX 3050/3060/3070/3080/3090, A100, A4000 | 8.0 / 8.6 / 8.7 | `ampere` | **Validado** (tensor cores) |
| Ada | RTX 4060/4070/4080/4090, L4, L40S | 8.9 | `ada` | Compila; sin probar en hardware |
| Blackwell | RTX 5060/5070/5080/5090 | 12.0 | `blackwell` | Compila; sin probar en hardware |
| Hopper | H100, H200 | 9.0 | `h100` | Compila; sin probar |
| Blackwell DC | B200 | 10.0 | `b200` | Compila; sin probar |

**No soportadas:** Maxwell (GTX 9xx) y Kepler — no tienen `dp4a.s32.s32`, que es
la instrucción base del proof-of-work. Pascal P100 (sm_60) tampoco: usa DP4A
recién desde sm_61.

> Las rutas DP4A (Pascal/Volta/Turing) son correctas pero mucho más lentas que
> la ruta con tensor cores. Como referencia medida: GTX 1660 SUPER ≈ 0.6 TH/s;
> RTX 3060 ≈ 33 TH/s con la geometría estándar.

---

## Requisitos

**Windows (recomendado)**

- GPU NVIDIA con driver reciente (probado con 617.42).
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/)
  con el workload *Desktop development with C++* (`cl.exe`, `cmake`, `ninja`).
- **CUDA Toolkit 12.8** (nvcc). CUDA 13 ya no compila sm_61/sm_70.
- [Rust](https://rustup.rs) (`cargo`).
- Python 3 (codegen de kernels).

**Linux** — `cargo`, `make`, `python3`, `git` y CUDA 12.x (`nvcc`).

---

## Compilar

### Windows

```powershell
.\build.ps1              # detecta tu GPU y compila su perfil
.\build.ps1 -All         # paquete completo: pascal, turing, volta, ampere, ada, blackwell
.\build.ps1 -Arch ampere # fuerza un perfil concreto
```

Genera `out/` listo para copiar a otra máquina:

```
out/
├── pearl-host.exe
├── pearl_mining_capi.dll
├── cudart64_12.dll
└── libs/
    ├── pascal/pearl_gemm_capi.dll
    ├── turing/pearl_gemm_capi.dll
    └── ...
```

`pearl-host.exe` se enlaza con CRT estático: no necesita el VC++ Redistributable.

### Linux / Docker (experimental)

```bash
PEARL_GEMM_ARCH=all ./build.sh        # mismo layout en ./out

docker build -t pearlhost .
docker run --gpus all -e PEARL_WALLET=prl1... pearlhost
```

---

## Uso

```powershell
.\out\pearl-host.exe --wallet prl1TU_DIRECCION --worker rig1
```

La wallet es **obligatoria** (o define `PEARL_WALLET`). Si no la pasás, el
minero no arranca: evita mandar shares a una dirección que no es la tuya.

| Opción | Descripción | Default |
|---|---|---|
| `--wallet <dir>` | wallet que recibe los pagos | — |
| `--worker <nombre>` | nombre del worker | `pearlhost` |
| `--host <host>` | pool | `br.pearl.herominers.com` |
| `--port <puerto>` | puerto | `1200` |
| `--gpu <índice>` | GPU a usar | `0` |
| `--slots <n>` | nonces por batch | `16` |
| `--lowmem` | geometría reducida (3–4 GB de VRAM) | auto |
| `--selfcheck` | verifica seeds GPU == host y sale | — |
| `--list-profiles` | muestra las DLLs y su ABI | — |
| `--dll <ruta>` | fuerza la DLL del motor | auto |
| `--mining-dll <ruta>` | fuerza la DLL de hashing | auto |
| `--cuda-bin <ruta>` | carpeta/DLL de `cudart` | auto |
| `--fee-percent <p>` | fee de desarrollo (0–5) | `1.0` |
| `--fee-wallet <dir>` | wallet del fee | la del autor |
| `--no-fee` | desactiva el fee | — |
| `-h`, `--help`, `-V`, `--version` | ayuda / versión | — |

### Ejemplos

```powershell
# PC de escritorio con RTX 3060
.\out\pearl-host.exe --wallet prl1... --worker pc

# Notebook / GPU de 4 GB
.\out\pearl-host.exe --wallet prl1... --lowmem

# Comprobar que el parche cert-v3 coincide con el pool (no mina)
.\out\pearl-host.exe --wallet prl1... --selfcheck

# Ver qué perfiles trae el paquete
.\out\pearl-host.exe --list-profiles
```

Salida típica:

```
  gpu    : 0 · NVIDIA GeForce RTX 3060 · 12.0 GiB · driver 617.42
  compute: 8.6 (sm_86, 28 SMs)
  motor  : ampere · ABI v3 · ...\libs\ampere\pearl_gemm_capi.dll
  geom   : m=8320 n=30720 k=32768 r=128 (standard) · VRAM ~3.7 GiB
  ✓ stratum autorizado
  job    : 00000003_2097152  height=134722  cert_version=3
  ▶ minando (job listo en 0.8 s)

  stats 3.90 it/s · 32.8 TH/s · ok=1 rech=0 · job ... h=134722
  gpu  : 110.2, 62, 45, 1920 (W,°C,%,MHz)
```

---

## Geometría y VRAM

La geometría (`m`, `n`, `k`, `r`) vive dentro del proof; el pool la reconstruye
desde los índices de filas y normaliza la dificultad por `r·k`. El minero elige
entre dos:

| Perfil | m | n | k | r | VRAM aprox. |
|---|---|---|---|---|---|
| `standard` | 8320 | 30720 | 32768 | 128 | ~3.7 GiB |
| `lowmem` | 4096 | 8192 | 32768 | 128 | ~1.3 GiB |

Se usa `lowmem` automáticamente si hay menos de 4.9 GiB libres, o si pasás
`--lowmem`. En GPUs de 6 GB conviene `--lowmem` si el escritorio consume VRAM.

---

## Fee de desarrollo

- **1 % del tiempo de minado**, en ventanas de 60 segundos.
- Se implementa cambiando temporalmente la wallet autorizada en el pool
  (`mining.authorize`) y volviendo a la tuya al terminar la ventana.
- La wallet del fee es **`prl1peyn72nhze69sju9g6nkjehqmtrdzgjqgg8excyp805uuch0thzqqyav93y`**.
- Si el pool rechaza el cambio de wallet, el fee se desactiva solo en esa
  sesión (nunca se pierden shares por el fee).
- Si tu wallet de pago es la misma que la del fee, el fee queda en 0 % efectivo.
- `--no-fee` lo desactiva por completo.

---

## Cómo funciona

1. **Stratum v2** (`br.pearl.herominers.com:1200`): `authorize`, `notify`
   (`job_id`, `header` de 76 B, `target` de 32 B BE, `height`, `cert_version`),
   `submit` (`plain_proof` = base64(gzip(bincode V1))).
2. **Commitment**: del `header` sale el `job_key`; de ahí las seeds de ruido.
   El host expande `B` y construye los árboles BLAKE3 keyed; la GPU calcula
   `A`, `AHash`, `BHash` y los commitments.
3. **Minado**: el kernel hace el GEMM int8 con ruido de bajo rango, hashea el
   transcript y compara contra el target escalado. Cuando un slot gana, el host
   regenera `A` del seed ganador, arma el `PlainProof` con los índices de tile
   y lo envía.
4. **Parche cert-v3**: la red mina con `cert_version=3` (salted). El motor lleva
   un parche propio (`ABI v3`) que aplica `SEED_SALT_A/B` a `bind_root`; con el
   commitment legacy **todos** los shares se rechazan. `--selfcheck` compara
   `CommitA/CommitB` de GPU contra el host y es la prueba rápida de que el
   parche está bien.

### Estructura

```
.
├── build.ps1                 # build Windows (host + DLLs por arquitectura)
├── build.sh                  # build Linux (experimental)
├── Dockerfile                # imagen con todos los perfiles
├── host/                     # host Rust (pearl-host)
│   └── src/{main,engine,gpu,stratum,share,proof,config,derive}.rs
└── native/
    ├── pearl-gemm/           # motor CUDA (CMake/Makefile por perfil)
    ├── pearl-blake3/         # BLAKE3 keyed-merkle
    └── pearl-mining-capi/    # C ABI de hashing (libpearl_mining_capi)
```

---

## Problemas comunes

| Síntoma | Causa / solución |
|---|---|
| `falta --wallet` | pasá `--wallet prl1...` o definí `PEARL_WALLET`. |
| `no encuentro el motor CUDA para el perfil 'x'` | compilá ese perfil: `.\build.ps1 -Arch x` (o `-All`). |
| `-209 cudaErrorNoKernelImageForDevice` | la DLL no trae código para tu GPU (perfil equivocado o cache CMake viejo). Recompilá con `-Arch` correcto; `build.ps1` limpia caches obsoletos solo. |
| `sin VRAM` / error al reservar | usá `--lowmem` y cerrá apps que usen la GPU. |
| `cudaGetDeviceCount: ...` | driver NVIDIA ausente o `cudart` no encontrado; pasá `--cuda-bin`. |
| Shares rechazados | revisá que `cert_version=3` y que `--selfcheck` dé `✓ seeds GPU == host`. |
| Se reconecta cada rato | firewall/ISP bloqueando el puerto 1200; probá otro pool. |

---

## Créditos

- Motor CUDA y derivación de seeds: implementación de referencia open source de
  **Akoya Pool** (kernels Ampere/Ada/Blackwell/Hopper, BLAKE3, transcript GEMM),
  con el parche `cert-v3` de este proyecto.
- [CUTLASS](https://github.com/NVIDIA/cutlass) — BSD-3-Clause (submódulo).
- Protocolo y validación de shares contrastados contra el pool HeroMiners.

El fee del 1 % mantiene el proyecto: si te sirve, dejalo activado.
