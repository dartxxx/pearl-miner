//! Derivación de seeds de Pearl (job_key + noise seeds).
//! Debe coincidir EXACTAMENTE con zk-pow/src/ffi/mine.rs y api/seed.rs (ISC).
//! Validado contra los vectores fijados de los tests de consenso de Pearl.

/// Salt de separación de dominio para A: blake3("pearl/cert-v3/noise-seed/A").
pub const SEED_SALT_A: [u8; 32] = [
    0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8,
    0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b,
];

/// Salt de separación de dominio para B: blake3("pearl/cert-v3/noise-seed/B").
pub const SEED_SALT_B: [u8; 32] = [
    0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf, 0x71, 0x8b, 0x3f, 0x4d, 0x4f,
    0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8, 0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9,
];

/// Cómo se derivan los noise seeds de las raíces Merkle (según cert_version).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedDerivation {
    /// Pre-V3: sin salting.
    Legacy,
    /// V3 (cert_version 3): raíces salteadas con SEED_SALT_A/B.
    Salted,
}

impl SeedDerivation {
    /// cert_version → derivación. HeroMiners envía cert_version=3 → Salted.
    pub fn from_cert_version(v: u32) -> Self {
        match v {
            3 => SeedDerivation::Salted,
            _ => SeedDerivation::Legacy,
        }
    }
}

/// `root || dim(u32 LE) || 28 zero bytes` — un solo bloque BLAKE3 de 64 B.
fn bind_message(root: &[u8; 32], dim: u32) -> [u8; 64] {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(root);
    msg[32..36].copy_from_slice(&dim.to_le_bytes());
    msg
}

/// V3: blake3(hash_a || pad32(m), key=SEED_SALT_A). Commitea el nº de filas m.
pub fn bind_root_a(hash_a: &[u8; 32], m: u32) -> [u8; 32] {
    *blake3::keyed_hash(&SEED_SALT_A, &bind_message(hash_a, m)).as_bytes()
}

/// V3: blake3(hash_b || pad32(n), key=SEED_SALT_B). Commitea el nº de columnas n.
pub fn bind_root_b(hash_b: &[u8; 32], n: u32) -> [u8; 32] {
    *blake3::keyed_hash(&SEED_SALT_B, &bind_message(hash_b, n)).as_bytes()
}

/// job_key = blake3(header_bytes || config_bytes). (mine.rs compute_job_key)
pub fn compute_job_key(header_bytes: &[u8], config_bytes: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(header_bytes);
    h.update(config_bytes);
    *h.finalize().as_bytes()
}

/// Noise seeds (b, a) a partir del job_key y las raíces Merkle crudas de A/B.
/// Debe coincidir con mine.rs compute_commitment_hash_with_offsets (sin MoE).
pub fn compute_noise_seeds(
    job_key: &[u8; 32],
    raw_hash_a: &[u8; 32],
    raw_hash_b: &[u8; 32],
    m: u32,
    n: u32,
    derivation: SeedDerivation,
) -> ([u8; 32], [u8; 32]) {
    let (hash_a, hash_b) = match derivation {
        SeedDerivation::Legacy => (*raw_hash_a, *raw_hash_b),
        SeedDerivation::Salted => (bind_root_a(raw_hash_a, m), bind_root_b(raw_hash_b, n)),
    };
    // sin MoE: hash_activations = hash_a
    let mut b_input = Vec::with_capacity(64);
    b_input.extend_from_slice(job_key);
    b_input.extend_from_slice(&hash_b);
    let b_noise_seed = *blake3::hash(&b_input).as_bytes();

    let mut a_input = Vec::with_capacity(64);
    a_input.extend_from_slice(&b_noise_seed);
    a_input.extend_from_slice(&hash_a);
    let a_noise_seed = *blake3::hash(&a_input).as_bytes();

    (b_noise_seed, a_noise_seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectores fijados de Pearl (api/seed.rs). job_key=[0x11;32], hash_a=[0xAA;32], hash_b=[0xBB;32].
    const JOB_KEY: [u8; 32] = [0x11u8; 32];
    const HASH_A: [u8; 32] = [0xAAu8; 32];
    const HASH_B: [u8; 32] = [0xBBu8; 32];

    #[test]
    fn salts_match_context_strings() {
        let a = *blake3::hash(b"pearl/cert-v3/noise-seed/A").as_bytes();
        let b = *blake3::hash(b"pearl/cert-v3/noise-seed/B").as_bytes();
        assert_eq!(a, SEED_SALT_A, "SEED_SALT_A changed — consensus break");
        assert_eq!(b, SEED_SALT_B, "SEED_SALT_B changed — consensus break");
    }

    #[test]
    fn legacy_pinned_vector() {
        let (b, a) = compute_noise_seeds(&JOB_KEY, &HASH_A, &HASH_B, 0, 0, SeedDerivation::Legacy);
        let exp_b = [
            0xad, 0xd6, 0xf7, 0xea, 0x5f, 0xee, 0xbf, 0x89, 0xc8, 0xa7, 0x7e, 0x2e, 0xbf, 0xa0, 0xd8, 0x24,
            0x42, 0xe7, 0xdb, 0xb0, 0x04, 0x6d, 0xbd, 0x48, 0x97, 0x18, 0x61, 0xd1, 0x2f, 0xcb, 0x01, 0x77,
        ];
        let exp_a = [
            0x48, 0x3b, 0x07, 0xb6, 0xf7, 0x31, 0x05, 0x03, 0x0b, 0x94, 0x82, 0x25, 0x5f, 0x37, 0x72, 0x3f,
            0x3f, 0xed, 0x69, 0xae, 0x91, 0x67, 0x24, 0xee, 0x82, 0x91, 0x84, 0x8b, 0x8c, 0x28, 0x79, 0x4b,
        ];
        assert_eq!(b, exp_b, "legacy b_noise_seed changed — consensus break");
        assert_eq!(a, exp_a, "legacy a_noise_seed changed — consensus break");
    }

    #[test]
    fn salted_pinned_vector() {
        // new_for_tests(192, 320, 256) -> m=192, n=320
        let (b, a) = compute_noise_seeds(&JOB_KEY, &HASH_A, &HASH_B, 192, 320, SeedDerivation::Salted);
        let exp_b = [
            0x60, 0xed, 0x9b, 0x73, 0xc5, 0xa9, 0x59, 0x9b, 0x20, 0x0b, 0x6c, 0xd5, 0x63, 0xe7, 0xf0, 0xd5,
            0xd9, 0xa6, 0x7d, 0x24, 0x02, 0xd8, 0x5f, 0xd4, 0xef, 0x96, 0x6c, 0x58, 0x00, 0x80, 0xd0, 0xe5,
        ];
        let exp_a = [
            0x30, 0x17, 0x84, 0x16, 0x80, 0x05, 0xec, 0x83, 0x3a, 0xb0, 0xaa, 0x60, 0x00, 0x6f, 0x7f, 0xe7,
            0xfa, 0xaa, 0x95, 0x30, 0x7d, 0x8c, 0x1f, 0xc6, 0x81, 0x9b, 0x2f, 0xfd, 0xd7, 0x17, 0xec, 0xcf,
        ];
        assert_eq!(b, exp_b, "salted b_noise_seed changed — consensus break");
        assert_eq!(a, exp_a, "salted a_noise_seed changed — consensus break");
    }
}
