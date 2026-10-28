//! PlainProof — formato bincode EXACTO de Pearl (espejo de zk-pow/src/ffi/plain_proof.rs).
//!
//! Reutiliza `pearl_blake3::MerkleProof` (mismo crate que la referencia =>
//! serialización idéntica). Validado con roundtrip byte a byte contra un
//! share real capturado de HeroMiners.

use pearl_blake3::MerkleProof;
use serde::{Deserialize, Serialize};

/// Prueba Merkle de una matriz (A o Bᵀ). `row_indices` = filas abiertas.
#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixMerkleProof {
    pub proof: MerkleProof,
    pub row_indices: Vec<usize>,
}

/// Parámetros MoE (denso = None; la softfork densa rechaza proofs con MoE).
#[derive(Clone, Serialize, Deserialize)]
pub struct MoEProofParams {
    pub e: u16,
    pub top_k: u16,
    pub expert_idx: u16,
    pub routing_end_offsets: Vec<u32>,
    pub inner_a_rows: Vec<usize>,
    pub routing_proof: MerkleProof,
}

/// El share tal cual lo espera `mining.submit` (bincode → gzip → base64).
///
/// Orden de campos crítico (bincode fixint, usize=u64 LE):
/// m, n, k, noise_rank, a, bt, moe: Option (tag u8 0=None).
#[derive(Clone, Serialize, Deserialize)]
pub struct PlainProof {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub noise_rank: usize,
    pub a: MatrixMerkleProof,
    pub bt: MatrixMerkleProof,
    pub moe: Option<MoEProofParams>,
}

impl PlainProof {
    /// Serializa al formato WIRE de HeroMiners: bincode V1 (sin tag de `moe`).
    /// Es exactamente lo que SRBMiner envía y la pool acepta (share real = V1).
    pub fn to_wire(&self) -> Vec<u8> {
        let mut bytes = bincode::serialize(self).expect("PlainProof serializa");
        if self.moe.is_none() {
            // Quitar el tag Option::None final que añade el layout V2 actual.
            if bytes.last() == Some(&0x00) {
                bytes.pop();
            }
        }
        bytes
    }

    /// Deserialización tolerante: acepta formato actual y legacy V1 (sin tag de `moe`).
    pub fn deserialize_compat(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).or_else(|_| {
            let mut padded = bytes.to_vec();
            padded.push(0x00); // tag Option::None
            bincode::deserialize(&padded)
                .map_err(|e| format!("no es PlainProof válido (V2 ni V1): {e}"))
        })
    }
}

/// Normaliza índices absolutos restando el primero (como hace list_to_pattern).
pub fn normalize_indices(indices: &[usize]) -> Vec<u32> {
    if indices.is_empty() {
        return vec![];
    }
    let base = indices[0];
    indices.iter().map(|&i| (i - base) as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../Pearl-RE/muestras/proof_1733536_6fbcffcc.bin"
    );

    #[test]
    #[ignore = "Falta el fixture real de HeroMiners en ./Pearl-RE/muestras/. \
Poner el archivo share real ahí y quitar el ignore para validar roundtrip."]
    fn decode_real_hero_miners_share() {
        let bytes = std::fs::read(SAMPLE).expect("share real de HeroMiners");
        let p = PlainProof::deserialize_compat(&bytes).expect("decodifica");

        // Geometría observada en el trabajo vivo (SRBMiner / HeroMiners):
        assert_eq!(p.m, 8322);
        assert_eq!(p.n, 30720);
        assert_eq!(p.k, 65536);
        assert_eq!(p.noise_rank, 128);
        assert!(p.moe.is_none(), "la softfork densa rechaza MoE");

        // Árboles: leaf_data coherente con índices (filas de k bytes, chunks de 1024).
        assert_eq!(p.a.proof.leaf_data.len(), p.a.proof.leaf_indices.len());
        assert_eq!(p.bt.proof.leaf_data.len(), p.bt.proof.leaf_indices.len());
        assert_eq!(p.a.proof.total_leaves, p.m * p.k / 1024);
        assert_eq!(p.bt.proof.total_leaves, p.n * p.k / 1024);
    }

    #[test]
    #[ignore = "Falta el fixture real de HeroMiners en ./Pearl-RE/muestras/. \
Poner el archivo share real ahí y quitar el ignore para validar roundtrip."]
    fn roundtrip_byte_identical() {
        let bytes = std::fs::read(SAMPLE).expect("share real");
        let p = PlainProof::deserialize_compat(&bytes).expect("decodifica");
        let re = p.to_wire();
        assert_eq!(re.len(), bytes.len(), "tamaño distinto");
        assert_eq!(re, bytes, "roundtrip no idéntico");
    }

    #[test]
    #[ignore = "Falta el fixture real de HeroMiners en ./Pearl-RE/muestras/. \
Poner el archivo share real aí y quitar el ignore para validar patterns."]
    fn patterns_match_live_hero_miners_job() {
        let bytes = std::fs::read(SAMPLE).expect("share real");
        let p = PlainProof::deserialize_compat(&bytes).expect("decodifica");

        // list_to_pattern: normalize respecto al primer índice.
        let rows = normalize_indices(&p.a.row_indices);
        let cols = normalize_indices(&p.bt.row_indices);

        // Geometría del trabajo VIVO de HeroMiners (del share real, no del test ISC):
        assert_eq!(
            rows,
            vec![0, 8, 32, 40, 64, 72, 96, 104, 128, 136],
            "rows_pattern inesperado"
        );
        assert_eq!(
            cols,
            vec![
                0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225
            ],
            "cols_pattern inesperado"
        );
    }
}
