//! PearlHost — minero de Pearl (PRL) para GPUs NVIDIA.
//!
//! Motor CUDA de Akoya (parche cert-v3 propio) + host Rust:
//! stratum v2, commitments BLAKE3, geometría adaptativa por VRAM.
//!
//! Fee de desarrollo: 1 % del tiempo de minado, configurable.

// Hay helpers de FFI/diagnóstico que solo usan los tests o herramientas
// externas; no los marcamos uno por uno.
#![allow(dead_code)]

mod config;
mod derive;
mod diag;
mod engine;
mod gpu;
mod proof;
mod share;
mod stratum;

use config::{MiningConfiguration, U256Le};
use std::collections::HashMap;
use std::io::Write;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Wallet del fee de desarrollo (1 % del tiempo de minado).
const FEE_WALLET: &str = "prl1peyn72nhze69sju9g6nkjehqmtrdzgjqgg8excyp805uuch0thzqqyav93y";
const DEFAULT_WORKER: &str = "pearlhost";
const DEFAULT_HOST: &str = "br.pearl.herominers.com";
const DEFAULT_PORT: u16 = 1200;
const AGENT: &str = concat!("pearl-host/", env!("CARGO_PKG_VERSION"));
const FEE_PERCENT_DEFAULT: f64 = 1.0;
/// Ventana de minado para el fee cuando toca pagar.
const FEE_WINDOW_SECS: f64 = 60.0;

// ─── Colores ANSI ────────────────────────────────────────────────────────────
const R: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";
const BLUE: &str = "\x1b[34m";
const BRIGHT_GREEN: &str = "\x1b[92m";
const BRIGHT_YELLOW: &str = "\x1b[93m";
const BRIGHT_RED: &str = "\x1b[91m";
const BRIGHT_CYAN: &str = "\x1b[96m";

#[cfg(windows)]
fn enable_vt() {
    use std::os::windows::io::AsRawHandle;
    let h = std::io::stdout().as_raw_handle();
    unsafe {
        extern "system" {
            fn GetConsoleMode(h: *mut std::ffi::c_void, m: *mut u32) -> i32;
            fn SetConsoleMode(h: *mut std::ffi::c_void, m: u32) -> i32;
        }
        let mut mode: u32 = 0;
        if GetConsoleMode(h as *mut _, &mut mode) != 0 {
            SetConsoleMode(h as *mut _, mode | 0x0004);
        }
    }
}
#[cfg(not(windows))]
fn enable_vt() {}

// ─── Argumentos ──────────────────────────────────────────────────────────────

struct Args {
    wallet: String,
    worker: String,
    host: String,
    port: u16,
    gpu: i32,
    slots: i32,
    gemm_dll: Option<std::path::PathBuf>,
    mining_dll: Option<std::path::PathBuf>,
    cuda_bin: Option<std::path::PathBuf>,
    lowmem: bool,
    selfcheck: bool,
    list_profiles: bool,
    fee_percent: f64,
    fee_wallet: String,
}

fn usage() {
    println!(
        r#"{BOLD}PearlHost {VERSION}{R} — minero Pearl (PRL) para NVIDIA

{BOLD}USO{R}
  pearl-host [opciones]

{BOLD}POOL{R}
  --wallet <dir>       wallet que recibe los pagos (obligatoria; o PEARL_WALLET)
  --worker <nombre>    nombre de worker            [pearlhost]
  --host <host>        host del pool               [br.pearl.herominers.com]
  --port <puerto>      puerto del pool             [1200]

{BOLD}GPU{R}
  --gpu <indice>       índice de la GPU            [0]
  --slots <n>          nonces por batch            [16]
  --lowmem             geometría reducida (GPUs de 3-4 GB)
  --dll <ruta>         forzar pearl_gemm_capi.dll
  --mining-dll <ruta>  forzar pearl_mining_capi.dll
  --cuda-bin <ruta>    carpeta o DLL de cudart
  --list-profiles      muestra los perfiles compilados y sale

{BOLD}DIAGNÓSTICO{R}
  --selfcheck          verifica seeds GPU==host y sale (0 = OK)
  -h, --help           esta ayuda
  -V, --version        versión

{BOLD}DEV FEE{R}
  --fee-percent <p>    porcentaje de tiempo para el fee [1.0]
  --fee-wallet <dir>   wallet del fee
  --no-fee             desactiva el fee (no recomendado en builds públicas)

{BOLD}EJEMPLOS{R}
  pearl-host --wallet prl1... --worker rig1
  pearl-host --wallet prl1... --gpu 0 --lowmem
  pearl-host --wallet prl1... --selfcheck
"#
    );
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        wallet: String::new(),
        worker: DEFAULT_WORKER.to_string(),
        host: DEFAULT_HOST.to_string(),
        port: DEFAULT_PORT,
        gpu: 0,
        slots: 16,
        gemm_dll: None,
        mining_dll: None,
        cuda_bin: None,
        lowmem: false,
        selfcheck: false,
        list_profiles: false,
        fee_percent: FEE_PERCENT_DEFAULT,
        fee_wallet: FEE_WALLET.to_string(),
    };
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = |i: &mut usize, name: &str| -> String {
            *i += 1;
            argv.get(*i)
                .cloned()
                .unwrap_or_else(|| {
                    eprintln!("{BRIGHT_RED}falta el valor de {name}{R}");
                    std::process::exit(2);
                })
        };
        match key {
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("pearl-host {VERSION}");
                std::process::exit(0);
            }
            "--wallet" => a.wallet = val(&mut i, key),
            "--worker" => a.worker = val(&mut i, key),
            "--host" => a.host = val(&mut i, key),
            "--port" => a.port = val(&mut i, key).parse().unwrap_or(DEFAULT_PORT),
            "--gpu" => a.gpu = val(&mut i, key).parse().unwrap_or(0),
            "--slots" => a.slots = val(&mut i, key).parse().unwrap_or(16).max(1),
            "--dll" => a.gemm_dll = Some(val(&mut i, key).into()),
            "--mining-dll" => a.mining_dll = Some(val(&mut i, key).into()),
            "--cuda-bin" => a.cuda_bin = Some(val(&mut i, key).into()),
            "--lowmem" => a.lowmem = true,
            "--selfcheck" => a.selfcheck = true,
            "--list-profiles" => a.list_profiles = true,
            "--fee-percent" => {
                a.fee_percent = val(&mut i, key).parse().unwrap_or(FEE_PERCENT_DEFAULT)
            }
            "--fee-wallet" => a.fee_wallet = val(&mut i, key),
            "--no-fee" => a.fee_percent = 0.0,
            other => {
                eprintln!("{BRIGHT_RED}opción desconocida: {other}{R}");
                eprintln!("usa --help");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    // Wallet de pago: --wallet o PEARL_WALLET. Sin wallet no se mina (evita
    // mandar los shares a una dirección que no es la tuya).
    if a.wallet.is_empty() {
        a.wallet = std::env::var("PEARL_WALLET").unwrap_or_default();
    }
    let needs_wallet = !a.list_profiles;
    if needs_wallet && a.wallet.is_empty() {
        eprintln!("{BRIGHT_RED}falta --wallet <tu_direccion_prl1...>{R}");
        eprintln!("{DIM}esa wallet recibe los pagos del pool.{R}");
        eprintln!("{DIM}ejemplo:  pearl-host --wallet prl1... --worker rig1{R}");
        std::process::exit(2);
    }
    if needs_wallet && (!a.wallet.starts_with("prl1") || a.wallet.len() < 40) {
        eprintln!("{BRIGHT_RED}wallet inválida: {}{R}", a.wallet);
        eprintln!("{DIM}las direcciones de Pearl empiezan por 'prl1'.{R}");
        std::process::exit(2);
    }

    if !(0.0..=5.0).contains(&a.fee_percent) {
        eprintln!("{BRIGHT_RED}--fee-percent debe estar entre 0 y 5{R}");
        std::process::exit(2);
    }
    a
}

// ─── Geometría ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Geometry {
    m: i32,
    n: i32,
    k: i32,
    r: i32,
    label: &'static str,
}

impl Geometry {
    fn vram_bytes(self) -> u64 {
        let (m, n, k, r) = (
            self.m as u64,
            self.n as u64,
            self.k as u64,
            self.r as u64,
        );
        // A + B + ApEA + BpEB
        let mut total = 2 * m * k + 2 * n * k;
        // workspace: ruido A (m*k*4) + transcript (~m*n/2048*4)
        total += m * k * 4;
        total += (m / 128) * (n / 256) * 256 * 16 * 4;
        // ruido auxiliar
        total += 2 * m * r + 2 * (k * r) * 2 + 2 * (n * r) * 2;
        total
    }
}

fn pick_geometry(free_vram: u64, lowmem: bool) -> Geometry {
    let gib = free_vram as f64 / (1024.0 * 1024.0 * 1024.0);
    if lowmem || gib < 4.9 {
        Geometry {
            m: 4096,
            n: 8192,
            k: 32768,
            r: 128,
            label: "lowmem",
        }
    } else {
        Geometry {
            m: 8320,
            n: 30720,
            k: 32768,
            r: 128,
            label: "standard",
        }
    }
}

// ─── Dev fee ─────────────────────────────────────────────────────────────────

enum FeeEvent {
    Enter,
    Leave,
}

struct DevFee {
    enabled: bool,
    percent: f64,
    wallet: String,
    owed: f64,
    until: Option<Instant>,
    paid: f64,
}

impl DevFee {
    fn new(enabled: bool, percent: f64, wallet: String) -> Self {
        Self {
            enabled: enabled && percent > 0.0,
            percent,
            wallet,
            owed: 0.0,
            until: None,
            paid: 0.0,
        }
    }

    fn in_fee(&self) -> bool {
        self.until.is_some()
    }

    /// Contabiliza `secs` de minado real y decide si toca cambiar de wallet.
    fn note(&mut self, secs: f64) -> Option<FeeEvent> {
        if !self.enabled {
            return None;
        }
        if let Some(until) = self.until {
            if Instant::now() >= until {
                self.until = None;
                self.paid += FEE_WINDOW_SECS;
                return Some(FeeEvent::Leave);
            }
            return None;
        }
        // Fracción f del tiempo total: f/(1-f) = p/100.
        self.owed += secs * (self.percent / (100.0 - self.percent));
        if self.owed >= FEE_WINDOW_SECS {
            self.owed -= FEE_WINDOW_SECS;
            self.until = Some(Instant::now() + Duration::from_secs_f64(FEE_WINDOW_SECS));
            return Some(FeeEvent::Enter);
        }
        None
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn mask_wallet(w: &str) -> String {
    if w.len() <= 14 {
        return w.to_string();
    }
    format!("{}…{}", &w[..10], &w[w.len() - 4..])
}

fn sigma_seed_of(job_key: &[u8; 32]) -> u64 {
    u64::from_le_bytes(job_key[0..8].try_into().unwrap())
}

fn bseed_of(job_key: &[u8; 32]) -> [u8; 32] {
    *blake3::keyed_hash(job_key, b"pearl-host/bseed/v1").as_bytes()
}

fn hex(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn print_profiles() {
    let found = gpu::installed_profiles();
    if found.is_empty() {
        println!("{DIM}no hay DLLs compiladas en libs/<perfil>/ ni en build-win/{R}");
        println!("{DIM}compílalas con:  .\\build.ps1 -All{R}");
        return;
    }

    // En Windows el loader resuelve las importaciones por nombre de módulo:
    // si cudart ya está cargado, pearl_gemm_capi.dll lo encuentra aunque no
    // esté en el PATH.
    let _cudart = gpu::find_cudart(None)
        .ok()
        .and_then(|p| unsafe { libloading::Library::new(&p).ok() });

    println!("{BOLD}Perfiles compilados:{R}");
    for (name, path) in found {
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let (profile, abi) = unsafe {
            match libloading::Library::new(&path) {
                Ok(lib) => {
                    let profile = lib
                        .get::<unsafe extern "C" fn() -> *const i8>(b"pearl_capi_build_profile")
                        .ok()
                        .map(|f| {
                            let p = f();
                            if p.is_null() {
                                "?".to_string()
                            } else {
                                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                            }
                        })
                        .unwrap_or_else(|| "?".into());
                    let abi = lib
                        .get::<unsafe extern "C" fn() -> i32>(b"pearl_capi_abi_version")
                        .ok()
                        .map(|f| f())
                        .unwrap_or(0);
                    (profile, abi)
                }
                Err(e) => (format!("error de carga: {e}"), 0),
            }
        };
        let ok = if abi == 3 {
            format!("{GREEN}✓{R}")
        } else {
            format!("{RED}✗{R}")
        };
        println!(
            "  {ok} {:<10} {:<10} {DIM}abi v{}{R}  {DIM}{:>6.2} MiB{R}  {}",
            name,
            profile,
            abi,
            size as f64 / 1048576.0,
            path.display()
        );
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    enable_vt();
    let args = parse_args();

    if args.list_profiles {
        print_profiles();
        return;
    }

    println!("{BOLD}{BRIGHT_CYAN}╔══════════════════════════════════════════════════════════╗{R}");
    println!("{BOLD}{BRIGHT_CYAN}║  PearlHost {VERSION:<6} · minero Pearl para NVIDIA           ║{R}");
    println!("{BOLD}{BRIGHT_CYAN}╚══════════════════════════════════════════════════════════╝{R}");
    println!();
    println!("  {DIM}pool   :{R} {BOLD}{}:{}{R}", args.host, args.port);
    println!("  {DIM}wallet :{R} {}", mask_wallet(&args.wallet));
    println!("  {DIM}worker :{R} {}", args.worker);

    // ── 1. GPU real ────────────────────────────────────────────────────────
    let cudart = match gpu::find_cudart(args.cuda_bin.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{BRIGHT_RED}{e}{R}");
            eprintln!("{DIM}instala el driver NVIDIA (incluye cudart) o pasa --cuda-bin{R}");
            std::process::exit(1);
        }
    };
    let probe = match gpu::Probe::load(&cudart) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{BRIGHT_RED}{e}{R}");
            std::process::exit(1);
        }
    };
    let info = match probe.query(args.gpu) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{BRIGHT_RED}FALLO detectando GPU: {e}{R}");
            std::process::exit(1);
        }
    };
    println!(
        "  {DIM}gpu    :{R} {} · {} · {:.1} GiB · driver {}",
        info.index,
        info.name,
        info.vram_gib(),
        info.driver
    );
    println!(
        "  {DIM}compute:{R} {} ({}, {} SMs)",
        info.cc(),
        info.sm(),
        info.sm_count
    );

    let profile = match gpu::profile_for(info.major, info.minor) {
        Some(p) => p,
        None => {
            eprintln!(
                "{BRIGHT_RED}FALLO: {}{R}",
                gpu::unsupported_reason(info.major, info.minor)
            );
            std::process::exit(1);
        }
    };

    // ── 2. DLLs ────────────────────────────────────────────────────────────
    let gemm_path = match gpu::find_engine_dll(profile, args.gemm_dll.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{BRIGHT_RED}{e}{R}");
            std::process::exit(1);
        }
    };
    let mining_path = match gpu::find_mining_dll(args.mining_dll.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{BRIGHT_RED}{e}{R}");
            std::process::exit(1);
        }
    };
    let libs = match engine::Libs::load(&cudart, &gemm_path, &mining_path) {
        Ok(l) => std::sync::Arc::new(l),
        Err(e) => {
            eprintln!("{BRIGHT_RED}FALLO cargando DLLs: {e}{R}");
            std::process::exit(1);
        }
    };

    let abi = unsafe { (libs.abi_version)() };
    let built = unsafe {
        std::ffi::CStr::from_ptr((libs.build_profile)())
            .to_string_lossy()
            .into_owned()
    };
    if abi != 3 {
        eprintln!(
            "{BRIGHT_RED}FALLO: la DLL es ABI v{abi}; se necesita la v3 (parche cert-v3){R}"
        );
        std::process::exit(1);
    }
    if !engine::supports(&libs, info.major, info.minor) {
        eprintln!(
            "{BRIGHT_RED}FALLO: la DLL '{built}' no soporta {}.{}.{R}",
            info.major, info.minor
        );
        eprintln!(
            "{DIM}compila el perfil correcto:  .\\build.ps1 -Arch {profile}{R}"
        );
        std::process::exit(1);
    }
    println!(
        "  {DIM}motor  :{R} {GREEN}{built}{R} · ABI v3 · {}",
        gemm_path.display()
    );

    // ── 3. Geometría ───────────────────────────────────────────────────────
    let free = if info.vram_free > 0 { info.vram_free } else { info.vram_total };
    let geo = pick_geometry(free, args.lowmem);
    let est = geo.vram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "  {DIM}geom   :{R} m={} n={} k={} r={} {DIM}({}){R} · VRAM ~{:.2} GiB libres {:.2}",
        geo.m,
        geo.n,
        geo.k,
        geo.r,
        geo.label,
        est,
        free as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    if est * 1.05 > free as f64 / (1024.0 * 1024.0 * 1024.0) {
        eprintln!("{BRIGHT_YELLOW}AVISO: la geometría puede no caber; usa --lowmem{R}");
    }

    // ── 4. Fee ─────────────────────────────────────────────────────────────
    let fee_same_wallet = args.fee_wallet == args.wallet;
    let fee_enabled = args.fee_percent > 0.0 && !fee_same_wallet;
    let mut fee = DevFee::new(fee_enabled, args.fee_percent, args.fee_wallet.clone());
    if args.fee_percent <= 0.0 {
        println!("  {DIM}fee    :{R} desactivado (--no-fee)");
    } else if fee_same_wallet {
        println!(
            "  {DIM}fee    :{R} {:.2}% → misma wallet de pago (0% efectivo)",
            args.fee_percent
        );
    } else {
        println!(
            "  {DIM}fee    :{R} {:.2}% → {}",
            args.fee_percent,
            mask_wallet(&args.fee_wallet)
        );
    }

    // ── 5. Motor ───────────────────────────────────────────────────────────
    println!();
    let mut eng = match engine::GpuEngine::new(
        libs.clone(),
        args.gpu,
        geo.m,
        geo.n,
        geo.k,
        geo.r,
        args.slots,
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{BRIGHT_RED}FALLO iniciando GPU: {e}{R}");
            eprintln!("{DIM}prueba con --lowmem si la VRAM es justa{R}");
            std::process::exit(1);
        }
    };
    println!(
        "  {GREEN}✓ VRAM reservada{R} (A {:.0} MiB, B {:.0} MiB, workspace ok)",
        geo.m as f64 * geo.k as f64 / 1048576.0,
        geo.n as f64 * geo.k as f64 / 1048576.0
    );

    // ── 6. Stratum ─────────────────────────────────────────────────────────
    let mut stratum = connect_authorized(&args, &args.wallet);
    println!("  {GREEN}✓ stratum autorizado{R}");
    let mut cur_job = match stratum.wait_first_job(Duration::from_secs(30)) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("{BRIGHT_RED}FALLO esperando job: {e}{R}");
            std::process::exit(1);
        }
    };
    println!(
        "  {BLUE}job    :{R} {}  height={}  cert_version={}",
        cur_job.job_id, cur_job.height, cur_job.cert_version
    );
    println!();

    let cfg = MiningConfiguration::mine_akoya_consumer();
    let factor = cfg.difficulty_adjustment_factor();
    let (m, n, k, r) = (
        geo.m as usize,
        geo.n as usize,
        geo.k as usize,
        geo.r as usize,
    );

    // ── 7. Bucle de minado ─────────────────────────────────────────────────
    let (b_tx, b_rx) = mpsc::channel::<(String, share::BState)>();
    let mut job_installed = false;
    let mut current_b: Option<share::BState> = None;
    let mut seed_lo: u64 = 0;
    let mut checked = false;
    let mut stats_start = Instant::now();
    let mut stats_iters: u64 = 0;
    let mut shares_ok: u64 = 0;
    let mut shares_rejected: u64 = 0;
    let mut submitted: HashMap<u64, bool> = HashMap::new(); // id → es_fee
    let mut pending_fee_auth: Option<u64> = None;

    loop {
        // ¿Job nuevo?
        if let Some(job) = stratum.latest_job() {
            if job.job_id != cur_job.job_id {
                println!(
                    "  {BLUE}job    :{R} {}  height={}  cert_version={} {DIM}(rotación){R}",
                    job.job_id, job.height, job.cert_version
                );
                cur_job = job;
                job_installed = false;
                current_b = None;
            }
        }

        if !job_installed {
            let job_key = cfg.job_key(&cur_job.header);
            let salted = if cur_job.cert_version == 3 { 1 } else { 0 };
            let pow = U256Le::from_be_bytes(&cur_job.target_be)
                .saturating_mul_u64(factor)
                .to_u32_le_words();
            let sigma_seed = sigma_seed_of(&job_key);
            let bseed = bseed_of(&job_key);

            let t_job = Instant::now();
            if let Err(e) = eng.install_job(&job_key, sigma_seed, &bseed, &pow, salted) {
                eprintln!("{BRIGHT_RED}FALLO install_job: {e}{R}");
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
            job_installed = true;

            // Autocomprobación V3 (una vez por proceso).
            if !checked {
                checked = true;
                match eng.dump_commit_state() {
                    Ok((ahash, bhash, dev_ca, dev_cb)) => {
                        let derivation = if salted == 1 {
                            derive::SeedDerivation::Salted
                        } else {
                            derive::SeedDerivation::Legacy
                        };
                        let (exp_b, exp_a) = derive::compute_noise_seeds(
                            &job_key, &ahash, &bhash, m as u32, n as u32, derivation,
                        );
                        let ok = dev_cb == exp_b && dev_ca == exp_a;
                        println!();
                        println!("  {DIM}[selfcheck] job_key = {}{R}", hex(&job_key));
                        println!("  {DIM}[selfcheck] AHash   = {}{R}", hex(&ahash));
                        println!("  {DIM}[selfcheck] BHash   = {}{R}", hex(&bhash));
                        println!("  {DIM}[selfcheck] CommitA gpu = {}{R}", hex(&dev_ca));
                        println!("  {DIM}[selfcheck] CommitA host= {}{R}", hex(&exp_a));
                        println!("  {DIM}[selfcheck] CommitB gpu = {}{R}", hex(&dev_cb));
                        println!("  {DIM}[selfcheck] CommitB host= {}{R}", hex(&exp_b));
                        if ok {
                            println!("  {GREEN}✓ seeds GPU == host (parche cert-v3 correcto){R}");
                        } else {
                            println!("  {BRIGHT_RED}✗ MISMATCH de seeds GPU vs host{R}");
                        }
                        println!();
                        if args.selfcheck {
                            std::process::exit(if ok { 0 } else { 1 });
                        }
                    }
                    Err(e) => {
                        eprintln!("  {RED}[selfcheck] falló: {e}{R}");
                        if args.selfcheck {
                            std::process::exit(1);
                        }
                    }
                }
            }

            let (tx, bseed_c, n_c, k_c, key_c) = (b_tx.clone(), bseed, n, k, job_key);
            let libs_bg = libs.clone();
            std::thread::spawn(move || {
                match engine::GpuEngine::host_expand_bseed(&libs_bg, &bseed_c, n_c, k_c) {
                    Ok(b_bytes) => {
                        let st = share::BState::build(&b_bytes, &key_c);
                        let _ = tx.send(("ok".into(), st));
                    }
                    Err(e) => eprintln!("{RED}[b-tree] expansión falló: {e}{R}"),
                }
            });
            println!(
                "  {GREEN}▶ minando{R} {DIM}(job listo en {:.1} s){R}",
                t_job.elapsed().as_secs_f64()
            );
            println!();
        }

        while let Ok((_tag, st)) = b_rx.try_recv() {
            current_b = Some(st);
        }

        let t0 = Instant::now();
        let winner = eng.mine_batch(seed_lo, args.slots);
        let batch_secs = t0.elapsed().as_secs_f64();
        seed_lo += args.slots as u64;
        stats_iters += args.slots as u64;

        match winner {
            Ok(Some((win_seed, header))) => {
                println!("{BOLD}{BRIGHT_YELLOW}  ★ seed={win_seed} — construyendo share…{R}");
                if current_b.is_none() {
                    for _ in 0..20 {
                        if let Ok((_t, st)) = b_rx.recv_timeout(Duration::from_millis(500)) {
                            current_b = Some(st);
                            break;
                        }
                    }
                }
                let Some(b_state) = current_b.as_ref() else {
                    eprintln!("  {RED}✗ share descartado: árbol de B no disponible{R}");
                    continue;
                };
                let a_bytes = match eng.regen_and_read_a(win_seed) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("  {RED}✗ regen A falló: {e}{R}");
                        continue;
                    }
                };
                let idx = engine::extract_indices(&header);
                let proof = match share::build_share(
                    &cfg.job_key(&cur_job.header),
                    &a_bytes,
                    b_state,
                    &idx.a_rows,
                    &idx.b_cols,
                    m,
                    n,
                    k,
                    r,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("  {RED}✗ build_share falló: {e}{R}");
                        continue;
                    }
                };
                let b64 = share::encode_share(&proof);
                let id = stratum.submit(&cur_job.job_id, &b64);
                submitted.insert(id, fee.in_fee());
                println!(
                    "  {MAGENTA}→ submit id={id} job={} proof={} KiB filas={:?} cols={}{R}",
                    cur_job.job_id,
                    b64.len() / 1024,
                    idx.a_rows,
                    idx.b_cols.len()
                );
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("{BRIGHT_RED}FALLO mine_batch: {e}{R}");
                std::thread::sleep(Duration::from_millis(500));
            }
        }

        // Fee: 1 % del tiempo, en ventanas de 60 s.
        match fee.note(batch_secs) {
            Some(FeeEvent::Enter) => {
                let id = stratum.authorize_async(&fee.wallet, &args.worker, AGENT);
                pending_fee_auth = Some(id);
                println!(
                    "  {DIM}[fee] {:.2}% → minando {:.0}s para {}{R}",
                    fee.percent,
                    FEE_WINDOW_SECS,
                    mask_wallet(&fee.wallet)
                );
            }
            Some(FeeEvent::Leave) => {
                let id = stratum.authorize_async(&args.wallet, &args.worker, AGENT);
                pending_fee_auth = Some(id);
                println!("  {DIM}[fee] vuelta a tu wallet{R}");
            }
            None => {}
        }

        // Respuestas del pool (shares + authorize del fee).
        while let Ok((rid, ok, err)) = stratum.resp_rx.try_recv() {
            if rid == 0 && !ok {
                eprintln!("{BRIGHT_RED}[stratum] desconectado: {err:?} — reconectando{R}");
                std::thread::sleep(Duration::from_secs(5));
                stratum = connect_authorized(&args, &args.wallet);
                cur_job = match stratum.wait_first_job(Duration::from_secs(30)) {
                    Ok(j) => j,
                    Err(e) => {
                        eprintln!("{BRIGHT_RED}sin job tras reconectar: {e}{R}");
                        std::process::exit(1);
                    }
                };
                job_installed = false;
                current_b = None;
                submitted.clear();
                fee.until = None;
                break;
            }
            if let Some(is_fee) = submitted.remove(&rid) {
                if ok {
                    shares_ok += 1;
                    let tag = if is_fee { " [fee]" } else { "" };
                    println!("  {BOLD}{BRIGHT_GREEN}✓ share ACEPTADO{tag} (id={rid}) total={shares_ok}{R}");
                } else {
                    shares_rejected += 1;
                    eprintln!("  {BRIGHT_RED}✗ share rechazado (id={rid}): {err:?}{R}");
                }
                continue;
            }
            if Some(rid) == pending_fee_auth {
                pending_fee_auth = None;
                if !ok {
                    eprintln!(
                        "  {BRIGHT_YELLOW}[fee] el pool rechazó el cambio de wallet: {err:?}{R}"
                    );
                    eprintln!("  {BRIGHT_YELLOW}[fee] fee desactivado para esta sesión{R}");
                    fee.enabled = false;
                    fee.until = None;
                }
            }
        }

        // Stats cada 5 s.
        let dt = stats_start.elapsed().as_secs_f64();
        if dt >= 5.0 {
            let ips = stats_iters as f64 / dt;
            let ths = ips * (m as f64 * n as f64 * k as f64) / 1e12;
            let smi = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=power.draw,temperature.gpu,fan.speed,clocks.current.graphics",
                    "--format=csv,noheader,nounits",
                    "-i",
                    &args.gpu.to_string(),
                ])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .unwrap_or_default();
            let smi = smi.trim();
            let fee_tag = if fee.in_fee() { " [FEE]" } else { "" };
            println!(
                "  {BOLD}{CYAN}stats{R} {ips:.2} it/s · {ths:.1} TH/s · ok={GREEN}{shares_ok}{R} rech={RED}{shares_rejected}{R} · job {} h={}{}",
                cur_job.job_id, cur_job.height, fee_tag
            );
            if !smi.is_empty() {
                println!("  {DIM}gpu  :{R} {smi} (W,°C,%,MHz)");
            }
            let _ = std::io::stdout().flush();
            stats_start = Instant::now();
            stats_iters = 0;
        }
    }
}

/// Conecta y autoriza, reintentando indefinidamente.
fn connect_authorized(args: &Args, wallet: &str) -> stratum::StratumClient {
    loop {
        match stratum::StratumClient::connect(&args.host, args.port) {
            Ok(c) => match c.authorize(wallet, &args.worker, AGENT) {
                Ok(()) => return c,
                Err(e) => {
                    eprintln!("{RED}authorize falló ({e}), reintento en 5 s…{R}");
                    std::thread::sleep(Duration::from_secs(5));
                }
            },
            Err(e) => {
                eprintln!("{RED}conexión falló ({e}), reintento en 5 s…{R}");
                std::thread::sleep(Duration::from_secs(5));
            }
        }
    }
}
