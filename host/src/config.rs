//! Configuración de minado — espejo exacto de zk-pow/src/api/{proof.rs,proof_utils.rs}.
//!
//! El pool reconstruye esta config desde mis row/col indices del proof
//! (`list_to_pattern`) y con ella calcula el job_key. Si mi puerto difiere
//! aunque sea un byte, el pool recalcula un job_key distinto y rechaza con
//! "Hash A mismatch". Validado contra el share real de HeroMiners.

use crate::derive::compute_job_key;

/// PeriodicPattern: 3 dims `(stride, length)` anidadas → 6 bytes serializados.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodicPattern {
    pub shape: [(u32, u32); 3],
}

pub const NUM_DIMS: usize = 3;

impl PeriodicPattern {
    /// Reconstruye el pattern desde una lista normalizada (empieza en 0,
    /// estrictamente creciente). Es EXACTAMENTE lo que el pool hace con
    /// los row_indices del proof.
    pub fn from_list(pattern: &[u32]) -> Result<Self, String> {
        if pattern.is_empty() {
            return Err("patrón vacío".into());
        }
        if !pattern.windows(2).all(|w| w[0] < w[1]) {
            return Err("patrón no estrictamente creciente".into());
        }
        if pattern[0] != 0 {
            return Err("patrón no empieza en 0".into());
        }

        let mut p: Vec<u32> = pattern.to_vec();
        let mut shape_vec: Vec<(u32, u32)> = Vec::new();

        while p.len() > 1 {
            let mut found = false;
            for period in 1..p.len() {
                if p.len() % period == 0 {
                    let s = p[period];
                    let is_periodic = (0..p.len() - period).all(|i| p[i] + s == p[i + period]);
                    if is_periodic {
                        shape_vec.push((s, (p.len() / period) as u32));
                        p.truncate(period);
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return Err("el patrón no es periódico".into());
            }
        }

        // Inverso + relleno a NUM_DIMS con (period, 1)
        shape_vec.reverse();
        let period = shape_vec.last().map_or(1, |&(s, l)| s * l);
        while shape_vec.len() < NUM_DIMS {
            shape_vec.push((period, 1));
        }

        let mut result = Self {
            shape: shape_vec.try_into().unwrap(),
        };
        // is_valid: roundtrip por serialización
        let bytes = result.to_bytes();
        let restored = Self::from_bytes(&bytes)?;
        if restored.shape != result.shape {
            return Err("patrón no canónico".into());
        }
        result.shape = restored.shape;
        Ok(result)
    }

    /// Serializa a 6 bytes: cada dim como (factor-1, length-1) con
    /// factor = stride / min_stride acumulado.
    pub fn to_bytes(&self) -> [u8; 2 * NUM_DIMS] {
        let mut data = [0u8; 2 * NUM_DIMS];
        let mut min_stride = 1u32;
        for (i, &(stride, length)) in self.shape.iter().enumerate() {
            let factor = stride / min_stride;
            data[2 * i] = (factor - 1) as u8;
            data[2 * i + 1] = (length - 1) as u8;
            min_stride = stride * length;
        }
        data
    }

    /// Parsea 6 bytes (inverso de to_bytes, con las mismas validaciones).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != 2 * NUM_DIMS {
            return Err(format!("esperaba {} bytes, llegó {}", 2 * NUM_DIMS, data.len()));
        }
        let mut shape = [(0u32, 0u32); NUM_DIMS];
        let mut min_stride = 1u32;
        let mut is_done = false;

        for (i, chunk) in data.chunks(2).enumerate() {
            let factor = 1 + (chunk[0] as u32);
            let length = 1 + (chunk[1] as u32);

            if length == 1 || is_done {
                if factor != 1 || length != 1 {
                    return Err("representación no canónica".into());
                }
                is_done = true;
            } else if factor <= 1 && min_stride != 1 {
                return Err("una sola stride no debe romperse".into());
            }

            let max_period = (1u64 << 24) / (factor as u64 * length as u64);
            if (min_stride as u64) > max_period {
                return Err("periodo > 2^24".into());
            }
            let stride = factor * min_stride;
            shape[i] = (stride, length);
            min_stride = stride * length;
        }

        Ok(Self { shape })
    }

    /// Índices generados (to_list de la referencia).
    pub fn to_list(&self) -> Vec<u32> {
        let mut res = vec![0u32];
        for &(stride, length) in &self.shape {
            let mut new_res = Vec::with_capacity(res.len() * length as usize);
            for i in 0..length {
                for &r in &res {
                    new_res.push(r + i * stride);
                }
            }
            res = new_res;
        }
        res
    }

    /// ¿Es válido un offset (tile base) para este patrón?
    pub fn offset_is_valid(&self, mut offset: u32) -> bool {
        for &(stride, length) in self.shape.iter().rev() {
            offset %= stride * length;
            if offset >= stride {
                return false;
            }
        }
        true
    }

    pub fn period(&self) -> u32 {
        let &(stride, length) = self.shape.last().unwrap();
        stride * length
    }
}

/// mma_type: solo Int7xInt7ToInt32 (= 0) existe en la cadena actual.
pub const MMA_INT7XINT7_TO_INT32: u16 = 0;

/// MiningConfiguration — 52 bytes serializados:
/// common_dim u32 LE | rank u16 LE | mma_type u16 LE | rows 6 | cols 6 | trailer 32.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiningConfiguration {
    pub common_dim: u32,
    pub rank: u16,
    pub mma_type: u16,
    pub rows_pattern: PeriodicPattern,
    pub cols_pattern: PeriodicPattern,
    /// Trailer: e(2) | top_k(2) | ceros(28). Denso = todo cero.
    pub trailer: [u8; 32],
}

impl MiningConfiguration {
    pub const SERIALIZED_SIZE: usize = 52;

    /// Config del trabajo vivo de HeroMiners (verificado con el share real):
    /// k=65536, rank=128, rows=[0,8,32,40,64,72,96,104,128,136],
    /// cols=[0,1,32,33,...,224,225].
    pub fn live_hero_miners() -> Self {
        let rows = PeriodicPattern::from_list(&[
            0, 8, 32, 40, 64, 72, 96, 104, 128, 136,
        ])
        .expect("rows_pattern vivo");
        let cols = PeriodicPattern::from_list(&[
            0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225,
        ])
        .expect("cols_pattern vivo");
        Self {
            common_dim: 65536,
            rank: 128,
            mma_type: MMA_INT7XINT7_TO_INT32,
            rows_pattern: rows,
            cols_pattern: cols,
            trailer: [0u8; 32],
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut bytes = Vec::with_capacity(Self::SERIALIZED_SIZE);
        bytes.extend_from_slice(&self.common_dim.to_le_bytes());
        bytes.extend_from_slice(&self.rank.to_le_bytes());
        bytes.extend_from_slice(&self.mma_type.to_le_bytes());
        bytes.extend_from_slice(&self.rows_pattern.to_bytes());
        bytes.extend_from_slice(&self.cols_pattern.to_bytes());
        bytes.extend_from_slice(&self.trailer);
        bytes.try_into().unwrap()
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != Self::SERIALIZED_SIZE {
            return Err(format!("esperaba 52 bytes, llegó {}", data.len()));
        }
        let common_dim = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let rank = u16::from_le_bytes(data[4..6].try_into().unwrap());
        let mma_type = u16::from_le_bytes(data[6..8].try_into().unwrap());
        if mma_type != MMA_INT7XINT7_TO_INT32 {
            return Err(format!("MMAType desconocido: {mma_type}"));
        }
        let rows_pattern = PeriodicPattern::from_bytes(&data[8..14])?;
        let cols_pattern = PeriodicPattern::from_bytes(&data[14..20])?;
        let mut trailer = [0u8; 32];
        trailer.copy_from_slice(&data[20..52]);
        if trailer[4..].iter().any(|&b| b != 0) {
            return Err("bytes reservados del trailer deben ser cero".into());
        }
        Ok(Self {
            common_dim,
            rank,
            mma_type,
            rows_pattern,
            cols_pattern,
            trailer,
        })
    }

    /// job_key = blake3(header_76B || config_52B) — 128 bytes = 2 bloques blake3.
    pub fn job_key(&self, header_76: &[u8; 76]) -> [u8; 32] {
        compute_job_key(header_76, &self.to_bytes())
    }

    /// Geometría que MINA nuestro host con el motor Akoya (kernel consumer
    /// CTA 128×256, exige m%128==0 y n%256==0):
    ///   m=8320 (65 tiles), n=30720 (120 tiles), k=32768, r=128.
    /// El thread ganador abre 2 filas {R,R+8} × 64 cols (pares stride 8):
    ///   rows_pattern=[0,8]            (h=2), cols=[0,1,8,9,…,248,249] (w=64).
    /// Valido contra sanity_checks: h·w=128∈[32,256]; (h+w)·k=2.16MB≤4MiB;
    /// k=32768: %64=0, ≥16r, ≥1024, ≤4r²; m,n≤2^24; offsets siempre validos
    /// (bases 128t mod 16=0<8; 256t mod 8=0<8). El pool infiere EXACTAMENTE
    /// esta config de nuestros row_indices (list_to_pattern) → job_key igual.
    pub fn mine_akoya_consumer() -> Self {
        let rows = PeriodicPattern::from_list(&[0, 8]).expect("rows [0,8]");
        let cols_list: Vec<u32> = {
            let mut v = Vec::with_capacity(64);
            for j in 0..32u32 {
                v.push(j * 8);
                v.push(j * 8 + 1);
            }
            v
        };
        let cols = PeriodicPattern::from_list(&cols_list).expect("cols pares stride 8");
        Self {
            common_dim: 32768,
            rank: 128,
            mma_type: MMA_INT7XINT7_TO_INT32,
            rows_pattern: rows,
            cols_pattern: cols,
            trailer: [0u8; 32],
        }
    }

    /// Geometría de minado (dims del problema que el host alloca).
    pub const MINE_M: u32 = 8320;
    pub const MINE_N: u32 = 30720;
    pub const MINE_K: u32 = 32768;
    pub const MINE_R: u32 = 128;

    /// Factor de ajuste de dificultad = h·w · dot_len (con rank=128, el
    /// penalizado y el simple coinciden: tile·(k/r)·128 = tile·k).
    /// El pool verifica hash ≤ target_pool × este factor.
    pub fn difficulty_adjustment_factor(&self) -> u64 {
        let h = self.rows_pattern.to_list().len() as u64;
        let w = self.cols_pattern.to_list().len() as u64;
        let k = self.common_dim as u64;
        h * w * k
    }
}

/// U256 little-endian limbs (limbs[0] = palabra menos significativa).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U256Le(pub [u32; 8]);

impl U256Le {
    pub const MAX: U256Le = U256Le([u32::MAX; 8]);

    /// Desde 32 bytes BIG-ENDIAN (como llegan en mining.notify.target).
    pub fn from_be_bytes(be: &[u8; 32]) -> Self {
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            let w = be[28 - i * 4..32 - i * 4].try_into().unwrap();
            limbs[i] = u32::from_be_bytes(w);
        }
        U256Le(limbs)
    }

    /// × factor con saturación en 2^256−1 (como Akoya GpuWorker.cs:1044).
    pub fn saturating_mul_u64(&self, factor: u64) -> Self {
        let mut out = [0u32; 8];
        let mut carry: u64 = 0;
        for i in 0..8 {
            let prod = (self.0[i] as u64) * factor + carry;
            out[i] = prod as u32;
            carry = prod >> 32;
        }
        if carry > 0 {
            return Self::MAX;
        }
        U256Le(out)
    }

    /// 8×u32 LE listo para subir a pow_target (el kernel compara MSW→LSW).
    pub fn to_u32_le_words(&self) -> [u32; 8] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Filas/cols ABSOLUTOS de un tile del share real (no normalizados).
    /// (idx[0] = tile base; el pool normaliza restando idx[0].)
    const LIVE_ROWS: [u32; 10] = [0, 8, 32, 40, 64, 72, 96, 104, 128, 136];
    const LIVE_COLS: [u16; 16] = [
        0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225,
    ];

    #[test]
    fn pattern_from_list_roundtrip() {
        let rows = PeriodicPattern::from_list(&LIVE_ROWS).unwrap();
        assert_eq!(rows.to_list(), LIVE_ROWS.to_vec(), "to_list ≠ entrada");

        let cols: Vec<u32> = LIVE_COLS.iter().map(|&c| c as u32).collect();
        let cp = PeriodicPattern::from_list(&cols).unwrap();
        assert_eq!(cp.to_list(), cols, "to_list ≠ entrada");
    }

    #[test]
    fn pattern_serialization_bytes() {
        let rows = PeriodicPattern::from_list(&LIVE_ROWS).unwrap();
        // dims (8,2),(32,5),(160,1) → (7,1),(1,4),(0,0)
        assert_eq!(rows.to_bytes(), [7, 1, 1, 4, 0, 0]);

        let cols: Vec<u32> = LIVE_COLS.iter().map(|&c| c as u32).collect();
        let cp = PeriodicPattern::from_list(&cols).unwrap();
        // dims (1,2),(32,8),(256,1) → (0,1),(15,7),(0,0)
        assert_eq!(cp.to_bytes(), [0, 1, 15, 7, 0, 0]);
    }

    #[test]
    fn config_roundtrip_and_shape() {
        let cfg = MiningConfiguration::live_hero_miners();
        let bytes = cfg.to_bytes();
        assert_eq!(bytes.len(), 52);
        assert_eq!(&bytes[0..4], &65536u32.to_le_bytes()); // common_dim
        assert_eq!(&bytes[4..6], &128u16.to_le_bytes()); // rank
        assert_eq!(&bytes[6..8], &[0, 0]); // mma_type = Int7xInt7ToInt32
        assert_eq!(&bytes[8..14], &[7, 1, 1, 4, 0, 0]); // rows
        assert_eq!(&bytes[14..20], &[0, 1, 15, 7, 0, 0]); // cols
        assert!(bytes[20..].iter().all(|&b| b == 0), "trailer denso = ceros");

        let back = MiningConfiguration::from_bytes(&bytes).unwrap();
        assert_eq!(back, cfg, "roundtrip de config");
    }

    #[test]
    fn mine_geometry_passes_pool_sanity_rules() {
        let cfg = MiningConfiguration::mine_akoya_consumer();

        // Reglas de public_params_sanity_check (zk-pow api/sanity_checks.rs)
        let k = cfg.common_dim as usize;
        let r = cfg.rank as usize;
        assert!(r.is_power_of_two() && (32..=1024).contains(&r), "rank");
        assert!(k <= (1 << 16) && k % 64 == 0 && k >= 1024, "k");
        assert!(k <= 4 * r * r && k >= 16 * r, "k vs 4r^2 / 16r");

        let h = cfg.rows_pattern.to_list().len();
        let w = cfg.cols_pattern.to_list().len();
        assert_eq!(h, 2);
        assert_eq!(w, 64);
        assert_eq!(cfg.rows_pattern.to_list(), vec![0, 8]);
        // Última col = (32 pares − 1)·8 + 1 = 249
        assert_eq!((w / 2 - 1) * 8 + 1, 249, "cols deben llegar a 249");
        assert_eq!(cfg.cols_pattern.to_list()[62..], [248, 249]);
        assert!(h % 2 == 0 && w % 2 == 0, "h,w % TILE_H(2)");
        assert!((32..=256).contains(&(h * w)), "h*w ∈ [32,256]");
        assert_eq!(h * w, 128);
        assert!((h + w) * k <= (1 << 22), "(h+w)*k ≤ 4 MiB");

        // Dims del motor: m%128==0, n%256==0 (kernel consumer CTA 128x256).
        assert_eq!(MiningConfiguration::MINE_M % 128, 0);
        assert_eq!(MiningConfiguration::MINE_N % 256, 0);

        // Offsets de tile siempre válidos (list_to_pattern → offset_is_valid).
        let m = MiningConfiguration::MINE_M;
        let n = MiningConfiguration::MINE_N;
        for tile_row in 0..(m / 128) {
            for r in 0..8u32 {
                let offset = tile_row * 128 + r;
                assert!(
                    cfg.rows_pattern.offset_is_valid(offset),
                    "rows offset {offset} invalido"
                );
                assert!(offset + 8 < m, "t_rows+rmax < m falla en {offset}");
            }
        }
        for tile_col in 0..(n / 256) {
            for c in [0u32, 2, 4, 6] {
                let offset = tile_col * 256 + c;
                assert!(
                    cfg.cols_pattern.offset_is_valid(offset),
                    "cols offset {offset} invalido"
                );
                assert!(offset + 249 < n, "t_cols+cmax < n falla en {offset}");
            }
        }
    }

    #[test]
    fn difficulty_factor_and_target_scaling() {
        let cfg = MiningConfiguration::mine_akoya_consumer();
        // h·w·k = 2·64·32768 = 4_194_304
        assert_eq!(cfg.difficulty_adjustment_factor(), 4_194_304);

        // Target real capturado de HeroMiners (32B BE, diff 2097152):
        let target_hex = "00000000000007fff80000000000000000000000000000000000000000000000";
        let mut be = [0u8; 32];
        for i in 0..32 {
            be[i] = u8::from_str_radix(&target_hex[2 * i..2 * i + 2], 16).unwrap();
        }
        let t = U256Le::from_be_bytes(&be);
        let scaled = t.saturating_mul_u64(cfg.difficulty_adjustment_factor());
        // No satura (target ~2^219 × 2^22 << 2^256)
        assert_ne!(scaled, U256Le::MAX);

        // Roundtrip: parseo BE → limbs LE.
        let zero = U256Le::from_be_bytes(&[0u8; 32]);
        assert_eq!(zero.0, [0u32; 8]);
        let max_be: [u8; 32] = [0xFF; 32];
        assert_eq!(U256Le::from_be_bytes(&max_be), U256Le::MAX);
        // Saturación: MAX × 2 satura.
        assert_eq!(U256Le::MAX.saturating_mul_u64(2), U256Le::MAX);
        // 1 × factor exacto.
        let mut one_be = [0u8; 32];
        one_be[31] = 1;
        let one = U256Le::from_be_bytes(&one_be);
        let s = one.saturating_mul_u64(4_194_304);
        assert_eq!(s.0[0], 4_194_304);
        assert_eq!(s.0[1..].iter().sum::<u32>(), 0);
    }

    #[test]
    fn job_key_is_stable_and_keyed() {
        let cfg = MiningConfiguration::live_hero_miners();
        let header = [0x42u8; 76];
        let k1 = cfg.job_key(&header);
        let k2 = cfg.job_key(&header);
        assert_eq!(k1, k2, "job_key debe ser determinista");
        let mut header2 = header;
        header2[0] ^= 1;
        assert_ne!(k1, cfg.job_key(&header2), "job_key debe depender del header");

        // Un bit de config cambia el job_key (si no, la pool y yo divergiríamos
        // en silencio solo si coincidiera por azar — ya validado en derive).
        let mut cfg2 = cfg.clone();
        cfg2.rank = 64;
        assert_ne!(k1, cfg2.job_key(&header));
    }
}
