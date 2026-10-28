//! Motor CUDA pearl_gemm_capi — FFI + orquestación de minado.
//!
//! Ciclo (espejo exacto del GpuWorker.cs de Akoya, validado en el código):
//!   1. GpuEngine::new: carga DLLs, alloca buffers + workspace (una vez).
//!   2. install_job: Key(job_key) + PowTarget(escalado) → throwaway A → AHash
//!      → install_B (expande B del bseed, V3-salted) → workspace_install_params.
//!   3. mine_batch: limpiar headers (host) → memset dSync → iter_batch →
//!      sync → escanear status de slots → ganador.
//!   4. Al ganar: regen A (lcg del seed ganador) → D2H → índices del header.

// `cuda_try!` / `gemm_try!` ya envuelven la llamada FFI en `unsafe` y se usan
// tanto dentro como fuera de bloques `unsafe`; el lint no aporta nada aquí.
#![allow(unused_unsafe)]

use std::ffi::{c_void, CStr};
use std::path::Path;

pub const TENSOR_HASH_THREADS: u32 = 128;
pub const TENSOR_HASH_STAGES: u32 = 2;
pub const TENSOR_HASH_LEAVES: u32 = 512;
pub const BM: i32 = 128;
pub const BN: i32 = 256;
pub const BK: i32 = 128;
pub const CM: i32 = 1;
pub const CN: i32 = 1;
pub const THROWAWAY_A_SEED_LO: u64 = 0xFFFF_FFFF_FFFF_FFFF;

// ─── Layout del HostSignalHeader (host_signal_header.hpp / HostSignalHeaderLayout.cs) ───
pub const HDR_STATUS: usize = 0; // i32: 0=idle 1=triggered
pub const HDR_TILE_COORD: usize = 40; // u32[3]
pub const HDR_NUM_REGS: usize = 64; // u16
pub const HDR_THREAD_ROWS: usize = 66; // u8[256]
pub const HDR_THREAD_COLS: usize = 322; // u8[256]
pub const HDR_MMA_SIZE: usize = 580; // i32[3]
pub const HDR_MMA_TILE: usize = 592; // i32[3]
pub const HDR_SIZE: usize = 640;

#[derive(Debug, Clone)]
pub struct TileIndices {
    pub a_rows: Vec<u32>,
    pub b_cols: Vec<u32>,
}

/// Extrae los índices del tile ganador (espejo de HostSignalHeaderLayout.ExtractIndices).
pub fn extract_indices(header: &[u8]) -> TileIndices {
    let g = |off: usize| u32::from_le_bytes(header[off..off + 4].try_into().unwrap());
    let tile_row = g(HDR_TILE_COORD);
    let tile_col = g(HDR_TILE_COORD + 4);
    let mma_tile_h = g(HDR_MMA_TILE) as u64;
    let mma_tile_w = g(HDR_MMA_TILE + 4) as u64;
    let num_regs =
        u16::from_le_bytes(header[HDR_NUM_REGS..HDR_NUM_REGS + 2].try_into().unwrap()) as usize;
    assert!(num_regs > 0 && num_regs <= 256, "num_regs fuera de rango");

    let mut rows = std::collections::BTreeSet::new();
    let mut cols = std::collections::BTreeSet::new();
    for i in 0..num_regs {
        rows.insert(header[HDR_THREAD_ROWS + i] as u64);
        cols.insert(header[HDR_THREAD_COLS + i] as u64);
    }
    let row_off = tile_row as u64 * mma_tile_h;
    let col_off = tile_col as u64 * mma_tile_w;
    TileIndices {
        a_rows: rows.iter().map(|&r| (row_off + r) as u32).collect(),
        b_cols: cols.iter().map(|&c| (col_off + c) as u32).collect(),
    }
}

#[repr(C)]
pub struct InstallBParams {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub r: i32,
    pub expand_bseed: i32,
    pub th_num_blocks: u32,
    pub th_threads: u32,
    pub th_stages: u32,
    pub th_leaves: u32,
    pub device_id: i32,
    pub bseed: *const u8,
    pub b: *mut c_void,
    pub b_hash: *mut c_void,
    pub key: *mut c_void,
    pub roots: *mut c_void,
    pub a_hash: *mut c_void,
    pub commit_a: *mut c_void,
    pub commit_b: *mut c_void,
    pub ear_k_major: *mut c_void,
    pub ebl_r_major: *mut c_void,
    pub ebl_k_major: *mut c_void,
    pub ebr: *mut c_void,
    pub ebr_fp16: *mut c_void,
    pub earx_bpeb: *mut c_void,
    pub bpeb: *mut c_void,
    pub workspace: *mut c_void,
    pub leaf_cvs: *mut c_void,
    pub salted: u32,
}

#[repr(C)]
pub struct WorkspaceParams {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub r: i32,
    pub b_m: i32,
    pub b_n: i32,
    pub b_k: i32,
    pub c_m: i32,
    pub c_n: i32,
    pub th_num_blocks: u32,
    pub th_threads: u32,
    pub th_stages: u32,
    pub th_leaves: u32,
    pub sigma_seed: u64,
    pub a: *mut c_void,
    pub b: *mut c_void,
    pub a_hash: *mut c_void,
    pub b_hash: *mut c_void,
    pub key: *mut c_void,
    pub roots: *mut c_void,
    pub commit_a: *mut c_void,
    pub commit_b: *mut c_void,
    pub eal: *mut c_void,
    pub eal_fp16: *mut c_void,
    pub ebr: *mut c_void,
    pub ebr_fp16: *mut c_void,
    pub ear_r_major: *mut c_void,
    pub ebl_r_major: *mut c_void,
    pub ear_k_major: *mut c_void,
    pub ebl_k_major: *mut c_void,
    pub ax_ebl_fp16: *mut c_void,
    pub earx_bpeb_fp16: *mut c_void,
    pub ap_ea: *mut c_void,
    pub bpeb: *mut c_void,
    pub a_scales: *mut c_void,
    pub b_scales: *mut c_void,
    pub c: *mut c_void,
    pub host_signal_sync: *mut c_void,
    pub pow_target: *mut c_void,
    pub pow_key: *mut c_void,
    pub salted: u32,
}

pub fn num_blocks(bytes: i64) -> u32 {
    let bpb = TENSOR_HASH_THREADS as i64 * 1024;
    ((bytes + bpb - 1) / bpb) as u32
}

fn cuda_msg(libs: &Libs, ctx: &str, rc: i32) -> String {
    unsafe {
        let p = (libs.cuda_get_error_string)(rc);
        let s = if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        if s.is_empty() {
            format!("{ctx}: cuda error {rc}")
        } else {
            format!("{ctx}: {s} ({rc})")
        }
    }
}

/// Pista legible para los códigos que devuelve la DLL del motor.
fn gemm_hint(rc: i32) -> &'static str {
    match rc {
        -2 => "excepción interna del motor",
        -10 => "dispositivo inválido",
        -11 => "matriz B demasiado grande (n*k > 4 GiB)",
        -22 => "k debe ser múltiplo de 32",
        -23 => "r debe dividir k",
        -30 => "sin VRAM para el scratch de ruido A",
        -31 => "fallo liberando workspace",
        -32 => "sin VRAM para el transcript",
        -33 => "fallo limpiando el transcript",
        209 | -209 => "cudaErrorNoKernelImageForDevice: esta DLL no trae código para tu GPU",
        34 | -34 => "cudaErrorMissingConfiguration",
        _ => "",
    }
}

macro_rules! cuda_try {
    ($libs:expr, $e:expr, $ctx:expr) => {{
        let rc = unsafe { $e };
        if rc != 0 {
            return Err(cuda_msg(&$libs, &$ctx, rc));
        }
    }};
}

macro_rules! gemm_try {
    ($e:expr, $ctx:expr) => {{
        let rc = unsafe { $e };
        if rc != 0 {
            let hint = gemm_hint(rc);
            return Err(if hint.is_empty() {
                format!("{}: gemm rc {} (0x{:x})", $ctx, rc, rc)
            } else {
                format!("{}: gemm rc {} — {}", $ctx, rc, hint)
            });
        }
    }};
}

/// FFI bruta de cudart + las dos DLLs. Cargada una vez por proceso.
pub struct Libs {
    #[allow(dead_code)]
    _cudart: libloading::Library,
    #[allow(dead_code)]
    _gemm: libloading::Library,
    #[allow(dead_code)]
    _mining: libloading::Library,
    pub cuda_malloc: unsafe extern "system" fn(*mut *mut c_void, usize) -> i32,
    pub cuda_free: unsafe extern "system" fn(*mut c_void) -> i32,
    pub cuda_malloc_host: unsafe extern "system" fn(*mut *mut c_void, usize) -> i32,
    pub cuda_free_host: unsafe extern "system" fn(*mut c_void) -> i32,
    pub cuda_memcpy_async:
        unsafe extern "system" fn(*mut c_void, *const c_void, usize, i32, *mut c_void) -> i32,
    pub cuda_memset_async: unsafe extern "system" fn(*mut c_void, i32, usize, *mut c_void) -> i32,
    pub cuda_stream_create: unsafe extern "system" fn(*mut *mut c_void) -> i32,
    pub cuda_stream_destroy: unsafe extern "system" fn(*mut c_void) -> i32,
    pub cuda_stream_synchronize: unsafe extern "system" fn(*mut c_void) -> i32,
    pub cuda_set_device: unsafe extern "system" fn(i32) -> i32,
    pub cuda_get_error_string: unsafe extern "system" fn(i32) -> *const i8,

    pub abi_version: unsafe extern "C" fn() -> i32,
    pub build_profile: unsafe extern "C" fn() -> *const i8,
    pub supports_sm: unsafe extern "C" fn(i32, i32) -> i32,
    pub get_sync_size: unsafe extern "C" fn() -> i32,
    pub get_header_size: unsafe extern "C" fn() -> i32,
    pub get_scratchpad_bytes: unsafe extern "C" fn(i64, i32) -> i64,
    pub tensor_hash: unsafe extern "C" fn(
        *const u8,
        u32,
        *mut u8,
        *const u8,
        u32,
        u32,
        u32,
        u32,
        *mut u8,
        i32,
        *mut c_void,
    ) -> i32,
    pub lcg_int7_fill:
        unsafe extern "C" fn(*mut c_void, i64, u64, u64, *mut c_void) -> i32,
    pub install_b: unsafe extern "C" fn(*const InstallBParams, *mut c_void) -> i32,
    pub workspace_alloc:
        unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, *mut *mut c_void, *mut c_void) -> i32,
    pub workspace_free: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    pub workspace_install_params:
        unsafe extern "C" fn(*mut c_void, *const WorkspaceParams) -> i32,
    pub iter_batch: unsafe extern "C" fn(
        *mut c_void,
        u64,
        *const *mut u8,
        i32,
        *mut c_void,
    ) -> i32,
    pub iter_batch_graph_prepare: unsafe extern "C" fn(
        *mut c_void,
        *const *mut u8,
        i32,
        *mut c_void,
    ) -> i32,
    pub iter_batch_graph_launch: unsafe extern "C" fn(
        *mut c_void,
        u64,
        *mut c_void,
    ) -> i32,
    pub bseed_expand_raw_mining: unsafe extern "C" fn(
        *const u8,
        usize,
        usize,
        *mut u8,
        usize,
        *mut *mut i8,
    ) -> i32,
    pub free_string_mining: unsafe extern "C" fn(*mut i8),
}

unsafe impl Send for Libs {}
unsafe impl Sync for Libs {}

impl Libs {
    pub fn load(cudart_path: &Path, gemm_path: &Path, mining_path: &Path) -> Result<Self, String> {
        unsafe {
            // cudart primero: satisface la dependencia de pearl_gemm_capi.dll.
            let cudart = libloading::Library::new(cudart_path)
                .map_err(|e| format!("cargar cudart {}: {e}", cudart_path.display()))?;
            let gemm = libloading::Library::new(gemm_path)
                .map_err(|e| format!("cargar gemm {}: {e}", gemm_path.display()))?;
            let mining = libloading::Library::new(mining_path)
                .map_err(|e| format!("cargar mining {}: {e}", mining_path.display()))?;

            macro_rules! sym {
                ($lib:expr, $name:ident) => {
                    $lib.get(stringify!($name).as_bytes())
                        .map_err(|e| format!("símbolo {}: {e}", stringify!($name)))?
                };
            }

            let l = Libs {
                cuda_malloc: *sym!(cudart, cudaMalloc),
                cuda_free: *sym!(cudart, cudaFree),
                cuda_malloc_host: *sym!(cudart, cudaMallocHost),
                cuda_free_host: *sym!(cudart, cudaFreeHost),
                cuda_memcpy_async: *sym!(cudart, cudaMemcpyAsync),
                cuda_memset_async: *sym!(cudart, cudaMemsetAsync),
                cuda_stream_create: *sym!(cudart, cudaStreamCreate),
                cuda_stream_destroy: *sym!(cudart, cudaStreamDestroy),
                cuda_stream_synchronize: *sym!(cudart, cudaStreamSynchronize),
                cuda_set_device: *sym!(cudart, cudaSetDevice),
                cuda_get_error_string: *sym!(cudart, cudaGetErrorString),
                abi_version: *sym!(gemm, pearl_capi_abi_version),
                build_profile: *sym!(gemm, pearl_capi_build_profile),
                supports_sm: *sym!(gemm, pearl_capi_supports_sm),
                get_sync_size: *sym!(gemm, pearl_capi_get_host_signal_sync_size),
                get_header_size: *sym!(gemm, pearl_capi_get_host_signal_header_size),
                get_scratchpad_bytes: *sym!(gemm, pearl_capi_get_required_scratchpad_bytes),
                tensor_hash: *sym!(gemm, pearl_capi_tensor_hash),
                lcg_int7_fill: *sym!(gemm, pearl_capi_lcg_int7_fill),
                install_b: *sym!(gemm, pearl_capi_install_B),
                workspace_alloc: *sym!(gemm, pearl_capi_workspace_alloc),
                workspace_free: *sym!(gemm, pearl_capi_workspace_free),
                workspace_install_params: *sym!(gemm, pearl_capi_workspace_install_params),
                iter_batch: *sym!(gemm, pearl_capi_iter_batch),
                iter_batch_graph_prepare: *sym!(gemm, pearl_capi_iter_batch_graph_prepare),
                iter_batch_graph_launch: *sym!(gemm, pearl_capi_iter_batch_graph_launch),
                bseed_expand_raw_mining: *sym!(mining, pearl_capi_bseed_expand_raw),
                free_string_mining: *sym!(mining, pearl_capi_free_string),
                _cudart: cudart,
                _gemm: gemm,
                _mining: mining,
            };
            Ok(l)
        }
    }
}

struct DevPtr(*mut c_void);
unsafe impl Send for DevPtr {}

/// Un "half" del worker: buffers GPU + workspace + slots de header.
pub struct GpuEngine {
    pub libs: std::sync::Arc<Libs>,
    pub device_id: i32,
    pub stream: *mut c_void,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub r: i32,
    pub slots: i32,

    pub a: *mut c_void,
    pub b: *mut c_void,
    pub key: *mut c_void,
    pub eal: *mut c_void,
    pub eal_fp16: *mut c_void,
    pub ear_r_major: *mut c_void,
    pub ear_k_major: *mut c_void,
    pub ebl_r_major: *mut c_void,
    pub ebl_k_major: *mut c_void,
    pub ebr: *mut c_void,
    pub ebr_fp16: *mut c_void,
    pub ax_ebl_fp16: *mut c_void,
    pub earx_bpeb: *mut c_void,
    pub ap_ea: *mut c_void,
    pub bpeb: *mut c_void,
    pub a_scales: *mut c_void,
    pub b_scales: *mut c_void,
    pub pow_target: *mut c_void,
    pub roots: *mut c_void,
    pub sync: *mut c_void,
    pub a_hash: *mut c_void,
    pub b_hash: *mut c_void,
    pub commit_a: *mut c_void,
    pub commit_b: *mut c_void,
    pub workspace: *mut c_void,

    // Pinned host
    pub host_headers: Vec<*mut u8>,
    pub host_header_ptrs: Vec<*mut u8>, // arreglo plano para iter_batch
    pub host_a: *mut u8,
    pub header_size: usize,
    pub b_a: i64,
    pub b_b: i64,

    pub sigma_seed: u64,
    pub job_installed: bool,
    pub graph_ready: bool,
}

unsafe impl Send for GpuEngine {}

pub fn supports(libs: &Libs, major: i32, minor: i32) -> bool {
    unsafe { (libs.supports_sm)(major, minor) == 1 }
}

impl GpuEngine {
    pub fn new(
        libs: std::sync::Arc<Libs>,
        device_id: i32,
        m: i32,
        n: i32,
        k: i32,
        r: i32,
        slots: i32,
    ) -> Result<Self, String> {
        let b_a = m as i64 * k as i64;
        let b_b = n as i64 * k as i64;
        let b_eal = m as i64 * r as i64;
        let b_ear_r = k as i64 * r as i64;
        let b_ear_k = r as i64 * k as i64;
        let b_ebr = n as i64 * r as i64;

        unsafe {
            cuda_try!(libs, (libs.cuda_set_device)(device_id), "set_device");
            let mut stream = std::ptr::null_mut();
            cuda_try!(libs, (libs.cuda_stream_create)(&mut stream), "stream_create");

            macro_rules! malloc {
                ($bytes:expr, $ctx:expr) => {{
                    let mut p = std::ptr::null_mut();
                    cuda_try!(libs, (libs.cuda_malloc)(&mut p, $bytes as usize), $ctx);
                    p
                }};
            }
            macro_rules! malloc_zero {
                ($bytes:expr, $ctx:expr) => {{
                    let p = malloc!($bytes, $ctx);
                    cuda_try!(
                        libs,
                        (libs.cuda_memset_async)(p, 0, $bytes as usize, stream),
                        concat!($ctx, "/zero")
                    );
                    p
                }};
            }
            macro_rules! malloc_host {
                ($bytes:expr, $ctx:expr) => {{
                    let mut p = std::ptr::null_mut();
                    cuda_try!(libs, (libs.cuda_malloc_host)(&mut p, $bytes as usize), $ctx);
                    p
                }};
            }

            // Escalas fp32 en 1.0 (pure-miner usa factor de cuantización 1).
            let alloc_fp32_ones = |count: i64, ctx: &str| -> Result<*mut c_void, String> {
                let mut p = std::ptr::null_mut();
                cuda_try!(libs, (libs.cuda_malloc)(&mut p, (count * 4) as usize), ctx);
                let ones: Vec<u8> = vec![0, 0, 0x80, 0x3F]
                    .iter()
                    .cycle()
                    .take((count * 4) as usize)
                    .copied()
                    .collect();
                cuda_try!(
                    libs,
                    (libs.cuda_memcpy_async)(
                        p,
                        ones.as_ptr() as *const c_void,
                        ones.len(),
                        1, // H2D
                        stream
                    ),
                    format!("{ctx}/h2d")
                );
                Ok(p)
            };

            let header_size = (libs.get_header_size)() as usize;
            assert_eq!(header_size, HDR_SIZE, "header size inesperado");
            let roots_bytes = (libs.get_scratchpad_bytes)(
                std::cmp::max(b_a, b_b),
                TENSOR_HASH_THREADS as i32,
            );

            let mut host_headers = Vec::with_capacity(slots as usize);
            let mut host_header_ptrs = Vec::with_capacity(slots as usize);
            for _ in 0..slots {
                let p = malloc_host!(header_size, "host_header") as *mut u8;
                std::ptr::write_bytes(p, 0, header_size);
                host_headers.push(p);
                host_header_ptrs.push(p);
            }
            let host_a = malloc_host!(b_a, "host_a") as *mut u8;

            let mut workspace = std::ptr::null_mut();
            // with_noise_A=1, with_noise_B=0: la referencia C# hace lo mismo.
            // Pedir noise_B gastaba n*k*4 bytes (~3.75 GiB con n=30720) que el
            // minero nunca usa (run_noising_B=false).
            gemm_try!(
                (libs.workspace_alloc)(m, n, k, r, 1, 0, &mut workspace, stream),
                "workspace_alloc"
            );

            let eng = GpuEngine {
                libs: libs.clone(),
                device_id,
                stream,
                m,
                n,
                k,
                r,
                slots,
                a: malloc!(b_a, "A"),
                b: malloc!(b_b, "B"),
                key: malloc!(32, "Key"),
                eal: malloc_zero!(b_eal, "EAL"),
                eal_fp16: malloc_zero!(b_eal * 2, "EAL_fp16"),
                ear_r_major: malloc_zero!(b_ear_r, "EAR_R"),
                ear_k_major: malloc_zero!(b_ear_k, "EAR_K"),
                ebl_r_major: malloc_zero!(b_ear_r, "EBL_R"),
                ebl_k_major: malloc_zero!(b_ear_k, "EBL_K"),
                ebr: malloc_zero!(b_ebr, "EBR"),
                ebr_fp16: malloc_zero!(b_ebr * 2, "EBR_fp16"),
                ax_ebl_fp16: malloc_zero!(b_eal * 2, "AxEBL_fp16"),
                earx_bpeb: malloc_zero!(b_ebr * 2, "EARxBpEB"),
                ap_ea: malloc_zero!(b_a, "ApEA"),
                bpeb: malloc_zero!(b_b, "BpEB"),
                a_scales: alloc_fp32_ones(m as i64, "A_scales")?,
                b_scales: alloc_fp32_ones(n as i64, "B_scales")?,
                pow_target: malloc!(32, "PowTarget"),
                roots: malloc_zero!(roots_bytes, "Roots"),
                sync: malloc_zero!((libs.get_sync_size)() as i64, "Sync"),
                a_hash: malloc_zero!(32, "AHash"),
                b_hash: malloc_zero!(32, "BHash"),
                commit_a: malloc_zero!(32, "CommitA"),
                commit_b: malloc_zero!(32, "CommitB"),
                workspace,
                host_headers,
                host_header_ptrs,
                host_a,
                header_size,
                b_a,
                b_b,
                sigma_seed: 0,
                job_installed: false,
                graph_ready: false,
            };
            cuda_try!(libs, (libs.cuda_stream_synchronize)(stream), "sync_alloc");
            Ok(eng)
        }
    }

    fn h2d(&self, dst: *mut c_void, src: &[u8], ctx: &str) -> Result<(), String> {
        unsafe {
            cuda_try!(
                self.libs,
                (self.libs.cuda_memcpy_async)(
                    dst,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    1,
                    self.stream
                ),
                ctx
            );
        }
        Ok(())
    }

    /// Instala un trabajo: job_key, sigma_seed, bseed (host 32B), pow_target
    /// (8×u32 LE ya escalado por el factor de dificultad) y `salted`
    /// (1 = cert_version 3 / V3; 0 = legacy).
    pub fn install_job(
        &mut self,
        job_key: &[u8; 32],
        sigma_seed: u64,
        bseed: &[u8; 32],
        pow_target_words: &[u32; 8],
        salted: u32,
    ) -> Result<(), String> {
        unsafe {
            self.h2d(self.key, job_key, "H2D Key")?;
            let target_bytes: Vec<u8> = pow_target_words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            self.h2d(self.pow_target, &target_bytes, "H2D PowTarget")?;
            self.sigma_seed = sigma_seed;

            // Throwaway A → AHash (install_B necesita un AHash para el commit).
            gemm_try!(
                (self.libs.lcg_int7_fill)(
                    self.a,
                    self.b_a,
                    THROWAWAY_A_SEED_LO,
                    sigma_seed,
                    self.stream
                ),
                "lcg throwaway"
            );
            gemm_try!(
                (self.libs.tensor_hash)(
                    self.a as *const u8,
                    self.b_a as u32,
                    self.a_hash as *mut u8,
                    self.key as *const u8,
                    num_blocks(self.b_a),
                    TENSOR_HASH_THREADS,
                    TENSOR_HASH_STAGES,
                    TENSOR_HASH_LEAVES,
                    self.roots as *mut u8,
                    self.device_id,
                    self.stream
                ),
                "tensor_hash throwaway A"
            );

            // install_B: expande B del bseed en GPU, V3-salted (nuestro parche).
            let p = InstallBParams {
                m: self.m,
                n: self.n,
                k: self.k,
                r: self.r,
                expand_bseed: 1,
                th_num_blocks: num_blocks(self.b_b),
                th_threads: TENSOR_HASH_THREADS,
                th_stages: TENSOR_HASH_STAGES,
                th_leaves: TENSOR_HASH_LEAVES,
                device_id: self.device_id,
                bseed: bseed.as_ptr(),
                b: self.b,
                b_hash: self.b_hash,
                key: self.key,
                roots: self.roots,
                a_hash: self.a_hash,
                commit_a: self.commit_a,
                commit_b: self.commit_b,
                ear_k_major: self.ear_k_major,
                ebl_r_major: self.ebl_r_major,
                ebl_k_major: self.ebl_k_major,
                ebr: self.ebr,
                ebr_fp16: self.ebr_fp16,
                earx_bpeb: self.earx_bpeb,
                bpeb: self.bpeb,
                workspace: self.workspace,
                leaf_cvs: std::ptr::null_mut(),
                salted, // cert_version 3 → 1 (HeroMiners); legacy → 0
            };
            gemm_try!((self.libs.install_b)(&p, self.stream), "install_B");

            // Constantes del workspace (hot path = pearl_capi_iter).
            let wp = WorkspaceParams {
                m: self.m,
                n: self.n,
                k: self.k,
                r: self.r,
                b_m: BM,
                b_n: BN,
                b_k: BK,
                c_m: CM,
                c_n: CN,
                th_num_blocks: num_blocks(self.b_a),
                th_threads: TENSOR_HASH_THREADS,
                th_stages: TENSOR_HASH_STAGES,
                th_leaves: TENSOR_HASH_LEAVES,
                sigma_seed,
                a: self.a,
                b: self.b,
                a_hash: self.a_hash,
                b_hash: self.b_hash,
                key: self.key,
                roots: self.roots,
                commit_a: self.commit_a,
                commit_b: self.commit_b,
                eal: self.eal,
                eal_fp16: self.eal_fp16,
                ebr: self.ebr,
                ebr_fp16: self.ebr_fp16,
                ear_r_major: self.ear_r_major,
                ebl_r_major: self.ebl_r_major,
                ear_k_major: self.ear_k_major,
                ebl_k_major: self.ebl_k_major,
                ax_ebl_fp16: self.ax_ebl_fp16,
                earx_bpeb_fp16: self.earx_bpeb,
                ap_ea: self.ap_ea,
                bpeb: self.bpeb,
                a_scales: self.a_scales,
                b_scales: self.b_scales,
                c: std::ptr::null_mut(), // pure-miner headless
                host_signal_sync: self.sync,
                pow_target: self.pow_target,
                pow_key: self.commit_a, // el kernel lee el CommitA per-iter
                salted,                // cert_version 3 → 1; legacy → 0
            };
            gemm_try!(
                (self.libs.workspace_install_params)(self.workspace, &wp),
                "workspace_install_params"
            );

            cuda_try!(
                self.libs,
                (self.libs.cuda_stream_synchronize)(self.stream),
                "sync_install"
            );
            self.job_installed = true;

            // CUDA graph: desactivado por ahora (problemas en sm_75 portable path).
            // El hot path usa iter_batch directo (GPU sigue al 100%).
            self.graph_ready = false;
        }
        Ok(())
    }

    /// Un batch de `count` nonces desde seed_lo. Devuelve (slot, winSeed, header)
    /// del ganador, o None.
    pub fn mine_batch(&mut self, seed_lo: u64, count: i32) -> Result<Option<(u64, Vec<u8>)>, String> {
        assert!(self.job_installed, "trabajo no instalado");
        unsafe {
            // INVARIANT 1: limpiar headers en host antes de lanzar.
            for h in &self.host_headers {
                std::ptr::write_bytes(*h, 0, self.header_size);
            }
            // INVARIANT 2: zero del dSync en el mismo stream (FIFO).
            cuda_try!(
                self.libs,
                (self.libs.cuda_memset_async)(self.sync, 0, 8, self.stream),
                "memset dSync"
            );
            gemm_try!(
                (self.libs.iter_batch)(
                    self.workspace,
                    seed_lo,
                    self.host_header_ptrs.as_ptr(),
                    count,
                    self.stream
                ),
                "iter_batch"
            );
            cuda_try!(
                self.libs,
                (self.libs.cuda_stream_synchronize)(self.stream),
                "sync batch"
            );

            // Escaneo: primer slot con status == 1.
            for (k, h) in self.host_headers.iter().enumerate() {
                let status = i32::from_le_bytes(std::slice::from_raw_parts(*h, 4).try_into().unwrap());
                if status == 1 {
                    let header =
                        std::slice::from_raw_parts(*h, self.header_size).to_vec();
                    return Ok(Some((seed_lo + k as u64, header)));
                }
            }
            Ok(None)
        }
    }

    /// Regenera A del seed ganador y lo baja a host (copia del pinned).
    pub fn regen_and_read_a(&mut self, win_seed: u64) -> Result<Vec<u8>, String> {
        unsafe {
            gemm_try!(
                (self.libs.lcg_int7_fill)(self.a, self.b_a, win_seed, self.sigma_seed, self.stream),
                "lcg regen A"
            );
            cuda_try!(
                self.libs,
                (self.libs.cuda_memcpy_async)(
                    self.host_a as *mut c_void,
                    self.a as *const c_void,
                    self.b_a as usize,
                    2, // D2H
                    self.stream
                ),
                "D2H A"
            );
            cuda_try!(
                self.libs,
                (self.libs.cuda_stream_synchronize)(self.stream),
                "sync D2H A"
            );
            Ok(std::slice::from_raw_parts(self.host_a, self.b_a as usize).to_vec())
        }
    }

    /// Dump de los 4 hashes de commitment (D2H) para autocomprobación V3.
    pub fn dump_commit_state(
        &self,
    ) -> Result<([u8; 32], [u8; 32], [u8; 32], [u8; 32]), String> {
        unsafe {
            let mut ahash = [0u8; 32];
            let mut bhash = [0u8; 32];
            let mut ca = [0u8; 32];
            let mut cb = [0u8; 32];
            macro_rules! d2h {
                ($dst:expr, $src:expr, $ctx:expr) => {
                    cuda_try!(
                        self.libs,
                        (self.libs.cuda_memcpy_async)(
                            $dst.as_mut_ptr() as *mut c_void,
                            $src as *const c_void,
                            32,
                            2,
                            self.stream
                        ),
                        $ctx
                    );
                };
            }
            d2h!(ahash, self.a_hash, "D2H AHash");
            d2h!(bhash, self.b_hash, "D2H BHash");
            d2h!(ca, self.commit_a, "D2H CommitA");
            d2h!(cb, self.commit_b, "D2H CommitB");
            cuda_try!(
                self.libs,
                (self.libs.cuda_stream_synchronize)(self.stream),
                "sync dump"
            );
            Ok((ahash, bhash, ca, cb))
        }
    }

    /// Expande el bseed en HOST (byte-idéntico a la GPU) para el árbol de B.
    pub fn host_expand_bseed(libs: &Libs, bseed: &[u8; 32], n: usize, k: usize) -> Result<Vec<u8>, String> {
        let total = n * k;
        let mut out = vec![0u8; total];
        let mut err: *mut i8 = std::ptr::null_mut();
        let rc = unsafe {
            (libs.bseed_expand_raw_mining)(
                bseed.as_ptr(),
                n,
                k,
                out.as_mut_ptr(),
                out.len(),
                &mut err,
            )
        };
        if rc != 0 {
            let msg = unsafe {
                if err.is_null() {
                    format!("rc {rc}")
                } else {
                    let s = std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned();
                    (libs.free_string_mining)(err);
                    s
                }
            };
            return Err(format!("bseed_expand_raw: {msg}"));
        }
        Ok(out)
    }
}

impl Drop for GpuEngine {
    fn drop(&mut self) {
        unsafe {
            let l = &self.libs;
            let _ = (l.cuda_stream_synchronize)(self.stream);
            let _ = (l.workspace_free)(self.workspace, self.stream);
            for p in [
                self.a, self.b, self.key, self.eal, self.eal_fp16, self.ear_r_major,
                self.ear_k_major, self.ebl_r_major, self.ebl_k_major, self.ebr,
                self.ebr_fp16, self.ax_ebl_fp16, self.earx_bpeb, self.ap_ea,
                self.bpeb, self.a_scales, self.b_scales, self.pow_target, self.roots,
                self.sync, self.a_hash, self.b_hash, self.commit_a, self.commit_b,
            ] {
                if !p.is_null() {
                    let _ = (l.cuda_free)(p);
                }
            }
            for h in &self.host_headers {
                if !(*h as *mut c_void).is_null() {
                    let _ = (l.cuda_free_host)(*h as *mut c_void);
                }
            }
            if !(self.host_a as *mut c_void).is_null() {
                let _ = (l.cuda_free_host)(self.host_a as *mut c_void);
            }
            let _ = (l.cuda_stream_destroy)(self.stream);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout_constants() {
        // Coherencia con HostSignalHeaderLayout.cs (offsets verificados ahí).
        assert_eq!(HDR_STATUS, 0);
        assert_eq!(HDR_TILE_COORD, 40);
        assert_eq!(HDR_NUM_REGS, 64);
        assert_eq!(HDR_THREAD_ROWS, 66);
        assert_eq!(HDR_THREAD_COLS, HDR_THREAD_ROWS + 256);
        assert_eq!(HDR_MMA_TILE, 592);
        assert_eq!(HDR_MMA_SIZE, 580);
    }

    #[test]
    fn num_blocks_exact() {
        // m*k = 8320*32768 = 272,629,760 → 2080 bloques exactos.
        assert_eq!(num_blocks(8320 * 32768), 2080);
        assert_eq!(num_blocks(30720 * 32768), 7680);
        // No múltiplo: redondeo hacia arriba.
        assert_eq!(num_blocks(131073), 2);
    }

    #[test]
    fn extract_indices_synthetic() {
        // Header sintético: tile (10, 20), mma tile 128×256, thread con
        // rows {3, 11} y cols {0,1,8,9}.
        let mut h = vec![0u8; HDR_SIZE];
        h[HDR_TILE_COORD..HDR_TILE_COORD + 4].copy_from_slice(&10u32.to_le_bytes());
        h[HDR_TILE_COORD + 4..HDR_TILE_COORD + 8].copy_from_slice(&20u32.to_le_bytes());
        h[HDR_MMA_TILE..HDR_MMA_TILE + 4].copy_from_slice(&128i32.to_le_bytes());
        h[HDR_MMA_TILE + 4..HDR_MMA_TILE + 8].copy_from_slice(&256i32.to_le_bytes());
        h[HDR_NUM_REGS..HDR_NUM_REGS + 2].copy_from_slice(&4u16.to_le_bytes());
        for (i, (r, c)) in [(3u8, 0u8), (11, 1), (3, 8), (11, 9)].iter().enumerate() {
            h[HDR_THREAD_ROWS + i] = *r;
            h[HDR_THREAD_COLS + i] = *c;
        }
        let idx = extract_indices(&h);
        assert_eq!(idx.a_rows, vec![1283, 1291]); // 10*128 + {3,11}
        assert_eq!(idx.b_cols, vec![5120, 5121, 5128, 5129]); // 20*256 + {0,1,8,9}
    }

    #[test]
    fn workspace_params_layout_matches_header() {
        // WorkspaceParams: i32×9 (0..36) + u32×4 (36..52) + pad 4 + sigma_seed(56)
        //   + 26 punteros (64..272) + salted u32 (272) + pad → size 280.
        assert_eq!(std::mem::size_of::<WorkspaceParams>(), 280);
        assert_eq!(std::mem::offset_of!(WorkspaceParams, sigma_seed), 56);
        assert_eq!(std::mem::offset_of!(WorkspaceParams, a), 64);
        assert_eq!(std::mem::offset_of!(WorkspaceParams, salted), 272);
        // InstallBParams: i32×10 (0..40) + bseed(40) + 16 punteros (48..176)
        //   + salted u32 (176) + pad → size 184.
        assert_eq!(std::mem::size_of::<InstallBParams>(), 184);
        assert_eq!(std::mem::offset_of!(InstallBParams, bseed), 40);
        assert_eq!(std::mem::offset_of!(InstallBParams, salted), 176);
    }
}
