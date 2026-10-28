//! Detección de NVIDIA, selección de perfil y localización de DLLs.
//!
//! El motor CUDA se compila una vez por arquitectura (`libs/<perfil>/`).
//! Aquí averiguamos la compute capability REAL de la tarjeta y elegimos la
//! DLL correcta, sin rutas absolutas ni suposiciones.

use std::ffi::CStr;
use std::path::{Path, PathBuf};

pub const PROFILES: &[&str] = &[
    "blackwell",
    "b200",
    "h100",
    "ada",
    "ampere",
    "turing",
    "volta",
    "pascal",
    "portable",
];

// cudaDeviceAttr values (cuda_runtime_api.h)
const ATTR_SM_COUNT: i32 = 16;
const ATTR_CC_MAJOR: i32 = 75;
const ATTR_CC_MINOR: i32 = 76;

#[cfg(windows)]
pub const CUDART_NAME: &str = "cudart64_12.dll";
#[cfg(not(windows))]
pub const CUDART_NAME: &str = "libcudart.so.12";

#[cfg(windows)]
pub const GEMM_NAME: &str = "pearl_gemm_capi.dll";
#[cfg(not(windows))]
pub const GEMM_NAME: &str = "libpearl_gemm_capi.so";

#[cfg(windows)]
pub const MINING_NAME: &str = "pearl_mining_capi.dll";
#[cfg(not(windows))]
pub const MINING_NAME: &str = "libpearl_mining_capi.so";

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: i32,
    pub name: String,
    pub major: i32,
    pub minor: i32,
    pub sm_count: i32,
    pub vram_total: u64,
    pub vram_free: u64,
    pub driver: String,
}

impl GpuInfo {
    pub fn cc(&self) -> String {
        format!("{}.{}", self.major, self.minor)
    }

    pub fn sm(&self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }

    pub fn vram_gib(&self) -> f64 {
        self.vram_total as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Perfil de compilación que corresponde a una compute capability.
pub fn profile_for(major: i32, minor: i32) -> Option<&'static str> {
    match (major, minor) {
        (12, _) => Some("blackwell"),
        (10, _) => Some("b200"),
        (9, 0) => Some("h100"),
        (8, 9) => Some("ada"),
        (8, 0) | (8, 6) | (8, 7) => Some("ampere"),
        (7, 5) => Some("turing"),
        (7, 0) => Some("volta"),
        (6, 1) => Some("pascal"),
        _ => None,
    }
}

/// Explicación legible para GPUs que no tienen perfil.
pub fn unsupported_reason(major: i32, minor: i32) -> String {
    match (major, minor) {
        (6, 0) => "Pascal P100 (sm_60) no tiene DP4A; usa una GTX 10xx (sm_61) o superior".into(),
        (6, 2) => "Pascal Tegra (sm_62) no está soportado por este motor".into(),
        (5, _) => "Maxwell (GTX 9xx) no tiene DP4A: no puede ejecutar pearlhash".into(),
        (3, _) | (2, _) => "Kepler/Fermi: sin soporte en CUDA 12 ni DP4A".into(),
        _ => format!("no hay perfil para compute capability {major}.{minor}"),
    }
}

fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn pick_file(mut candidates: Vec<PathBuf>, what: &str) -> Result<PathBuf, String> {
    let seen = candidates.clone();
    if let Some(p) = candidates.drain(..).find(|p| p.is_file()) {
        return Ok(p);
    }
    let mut msg = format!("no encuentro {what}. Rutas probadas:\n");
    for p in seen {
        msg.push_str(&format!("  - {}\n", p.display()));
    }
    Err(msg)
}

/// Localiza `cudart64_12.dll` (runtime CUDA). No requiere CUDA Toolkit si el
/// driver expone la DLL en el PATH del sistema.
pub fn find_cudart(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let mut c: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        if p.is_dir() {
            c.push(p.join(CUDART_NAME));
        } else {
            c.push(p.to_path_buf());
        }
    }
    if let Some(p) = env_path("PEARL_CUDA_BIN") {
        c.push(p.join(CUDART_NAME));
    }
    let exe = exe_dir();
    c.push(exe.join(CUDART_NAME));
    // Instalaciones estándar del toolkit (v12.x, la más nueva primero).
    for root in [
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
        PathBuf::from("/usr/local/cuda"),
    ] {
        if let Ok(entries) = std::fs::read_dir(&root) {
            let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            dirs.sort();
            dirs.reverse();
            for d in dirs {
                c.push(d.join("bin").join(CUDART_NAME));
                c.push(d.join("lib64").join(CUDART_NAME));
            }
        }
    }
    if let Some(cuda_path) = env_path("CUDA_PATH") {
        c.push(cuda_path.join("bin").join(CUDART_NAME));
    }
    // Último recurso: que lo resuelva el loader del sistema.
    c.push(PathBuf::from(CUDART_NAME));
    pick_file(c, "el runtime CUDA (cudart)")
}

/// Localiza la DLL del motor para el perfil elegido.
pub fn find_engine_dll(profile: &str, explicit: Option<&Path>) -> Result<PathBuf, String> {
    let mut c: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        if p.is_dir() {
            c.push(p.join(GEMM_NAME));
        } else {
            c.push(p.to_path_buf());
        }
    }
    if let Some(p) = env_path("PEARL_GEMM_DLL") {
        c.push(p);
    }
    let exe = exe_dir();
    let here = cwd();
    c.push(exe.join("libs").join(profile).join(GEMM_NAME));
    c.push(exe.join(GEMM_NAME));
    c.push(
        here.join("libs")
            .join(profile)
            .join(GEMM_NAME),
    );
    c.push(here.join(GEMM_NAME));
    c.push(
        here.join("native")
            .join("pearl-gemm")
            .join("csrc")
            .join("capi")
            .join("build-win")
            .join(profile)
            .join(GEMM_NAME),
    );
    pick_file(
        c,
        &format!(
            "el motor CUDA para el perfil '{profile}' ({GEMM_NAME}). \
             Compílalo con:  .\\build.ps1 -Arch {profile}   (o usa -All)"
        ),
    )
}

/// Localiza la DLL de BLAKE3/merkle.
pub fn find_mining_dll(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let mut c: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        if p.is_dir() {
            c.push(p.join(MINING_NAME));
        } else {
            c.push(p.to_path_buf());
        }
    }
    if let Some(p) = env_path("PEARL_MINING_DLL") {
        c.push(p);
    }
    let exe = exe_dir();
    let here = cwd();
    c.push(exe.join(MINING_NAME));
    c.push(here.join(MINING_NAME));
    c.push(here.join("native").join("target").join("release").join(MINING_NAME));
    c.push(exe.join("native").join("target").join("release").join(MINING_NAME));
    pick_file(c, &format!("la librería de hashing ({MINING_NAME})"))
}

/// Perfiles con DLL presente en disco (para `--list-profiles`).
/// Prioriza `libs/` (paquete listo); solo mira `build-win/` si no hay nada.
pub fn installed_profiles() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let mut roots = vec![exe_dir().join("libs"), cwd().join("libs")];
    for root in &roots {
        collect_profiles(root, &mut out);
    }
    if out.is_empty() {
        roots.clear();
        roots.push(
            cwd().join("native")
                .join("pearl-gemm")
                .join("csrc")
                .join("capi")
                .join("build-win"),
        );
        for root in &roots {
            collect_profiles(root, &mut out);
        }
    }
    out.sort();
    out
}

fn collect_profiles(root: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for e in entries.flatten() {
        let d = e.path();
        if !d.is_dir() {
            continue;
        }
        let dll = d.join(GEMM_NAME);
        if dll.is_file() {
            let name = d
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if !out.iter().any(|(n, _): &(String, PathBuf)| n == &name) {
                out.push((name, dll));
            }
        }
    }
}

/// Sonda mínima sobre cudart: no depende de la DLL del motor.
pub struct Probe {
    lib: libloading::Library,
    pub set_device: unsafe extern "system" fn(i32) -> i32,
    pub get_count: unsafe extern "system" fn(*mut i32) -> i32,
    pub get_attr: unsafe extern "system" fn(*mut i32, i32, i32) -> i32,
    pub mem_info: unsafe extern "system" fn(*mut usize, *mut usize) -> i32,
    pub error_name: unsafe extern "system" fn(i32) -> *const i8,
}

impl Probe {
    pub fn load(cudart: &Path) -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new(cudart)
                .map_err(|e| format!("cargar cudart {}: {e}", cudart.display()))?;
            macro_rules! s {
                ($n:ident) => {
                    *lib.get(stringify!($n).as_bytes())
                        .map_err(|e| format!("cudart sin {}: {e}", stringify!($n)))?
                };
            }
            Ok(Self {
                set_device: s!(cudaSetDevice),
                get_count: s!(cudaGetDeviceCount),
                get_attr: s!(cudaDeviceGetAttribute),
                mem_info: s!(cudaMemGetInfo),
                error_name: s!(cudaGetErrorName),
                lib,
            })
        }
    }

    pub fn error_name(&self, code: i32) -> String {
        unsafe {
            let p = (self.error_name)(code);
            if p.is_null() {
                format!("cuda error {code}")
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    pub fn device_count(&self) -> Result<i32, String> {
        let mut n = 0i32;
        let rc = unsafe { (self.get_count)(&mut n) };
        if rc != 0 {
            return Err(format!(
                "cudaGetDeviceCount: {} ({rc}). ¿Driver NVIDIA instalado?",
                self.error_name(rc)
            ));
        }
        Ok(n)
    }

    fn attr(&self, dev: i32, attr: i32) -> Result<i32, String> {
        let mut v = 0i32;
        let rc = unsafe { (self.get_attr)(&mut v, attr, dev) };
        if rc != 0 {
            return Err(format!(
                "cudaDeviceGetAttribute({attr}): {} ({rc})",
                self.error_name(rc)
            ));
        }
        Ok(v)
    }

    pub fn query(&self, index: i32) -> Result<GpuInfo, String> {
        let count = self.device_count()?;
        if count == 0 {
            return Err("no hay GPUs NVIDIA visibles".into());
        }
        if index < 0 || index >= count {
            return Err(format!("--gpu {index} fuera de rango (hay {count} GPU(s))"));
        }
        let rc = unsafe { (self.set_device)(index) };
        if rc != 0 {
            return Err(format!("cudaSetDevice({index}): {}", self.error_name(rc)));
        }
        let major = self.attr(index, ATTR_CC_MAJOR)?;
        let minor = self.attr(index, ATTR_CC_MINOR)?;
        let sm_count = self.attr(index, ATTR_SM_COUNT).unwrap_or(0);
        let (mut free, mut total) = (0usize, 0usize);
        let rc = unsafe { (self.mem_info)(&mut free, &mut total) };
        if rc != 0 {
            free = 0;
            total = 0;
        }
        let (name, driver) = nvidia_smi(index).unwrap_or_else(|| ("NVIDIA GPU".into(), "?".into()));
        Ok(GpuInfo {
            index,
            name,
            major,
            minor,
            sm_count,
            vram_total: total as u64,
            vram_free: free as u64,
            driver,
        })
    }
}

/// Nombre y driver vía nvidia-smi (opcional; no es crítico).
fn nvidia_smi(index: i32) -> Option<(String, String)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            &index.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let line = s.lines().next()?.trim();
    let mut it = line.split(',').map(|x| x.trim().to_string());
    let name = it.next()?;
    let driver = it.next().unwrap_or_else(|| "?".into());
    Some((name, driver))
}

/// Error de una función C de cudart con nombre legible.
pub fn cuda_error(probe: &Probe, ctx: &str, rc: i32) -> String {
    format!("{ctx}: {} ({rc})", probe.error_name(rc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfiles_por_compute_capability() {
        assert_eq!(profile_for(6, 1), Some("pascal"));
        assert_eq!(profile_for(7, 0), Some("volta"));
        assert_eq!(profile_for(7, 5), Some("turing"));
        assert_eq!(profile_for(8, 0), Some("ampere"));
        assert_eq!(profile_for(8, 6), Some("ampere"));
        assert_eq!(profile_for(8, 7), Some("ampere"));
        assert_eq!(profile_for(8, 9), Some("ada"));
        assert_eq!(profile_for(9, 0), Some("h100"));
        assert_eq!(profile_for(10, 0), Some("b200"));
        assert_eq!(profile_for(12, 0), Some("blackwell"));
    }

    #[test]
    fn gpus_sin_dp4a_no_tienen_perfil() {
        assert_eq!(profile_for(5, 2), None, "Maxwell no tiene DP4A");
        assert_eq!(profile_for(6, 0), None, "P100 (sm_60) no tiene DP4A");
        assert_eq!(profile_for(3, 7), None, "Kepler no está soportado");
        assert!(unsupported_reason(5, 2).contains("DP4A"));
        assert!(unsupported_reason(6, 0).contains("DP4A"));
    }
}
