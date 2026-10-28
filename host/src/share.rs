//! Construcción del share: árboles Merkle A/B (host) + PlainProof + gzip/base64.
//!
//! Propiedad clave (validada en test): la raíz de `pearl_blake3::MerkleTree`
//! sobre el buffer padeado con key=job_key ES `blake3::keyed_hash(job_key,
//! padded)` — exactamente lo que el `tensor_hash` de la GPU calcula como
//! AHash/BHash. Por eso los proofs construidos aquí enlazan con las raíces
//! que el kernel ya comprometió en el commitment (V3-salted).

use base64::Engine as _;
use pearl_blake3::{pad_to_chunk_boundary, MerkleTree};

use crate::proof::{MatrixMerkleProof, PlainProof};

/// Estado de B cacheado por job (expansión host + árbol).
pub struct BState {
    pub tree: MerkleTree,
    pub padded_len: usize,
}

impl BState {
    /// `b_bytes` = n*k crudos (de pearl_capi_bseed_expand_raw en host).
    pub fn build(b_bytes: &[u8], job_key: &[u8; 32]) -> Self {
        let padded = pad_to_chunk_boundary(b_bytes);
        let padded_len = padded.len();
        let tree = MerkleTree::new(&padded, *job_key);
        BState { tree, padded_len }
    }
}

/// Arma el PlainProof de un trigger.
///
/// - `a_bytes`  = matriz A cruda (m*k) regenerada con lcg del seed ganador.
/// - `a_rows`/`b_cols` = índices ABSOLUTOS extraídos del HostSignalHeader.
#[allow(clippy::too_many_arguments)]
pub fn build_share(
    job_key: &[u8; 32],
    a_bytes: &[u8],
    b_state: &BState,
    a_rows: &[u32],
    b_cols: &[u32],
    m: usize,
    n: usize,
    k: usize,
    r: usize,
) -> Result<PlainProof, String> {
    let a_rows_us: Vec<usize> = a_rows.iter().map(|&x| x as usize).collect();
    let b_cols_us: Vec<usize> = b_cols.iter().map(|&x| x as usize).collect();

    // A: pad + árbol (por trigger; ~m*k de blake3 con rayon).
    let a_padded = pad_to_chunk_boundary(a_bytes);
    let a_tree = MerkleTree::new(&a_padded, *job_key);
    let a_leaf_idx = MerkleTree::compute_leaf_indices_from_rows(&a_rows_us, (m, k));
    // Guarda contra "leaf index out of bounds" (t_rows+rmax < m debe cumplirse).
    if let Some(&bad) = a_leaf_idx.iter().find(|&&i| i >= a_tree.num_leaves()) {
        return Err(format!(
            "A: leaf idx {bad} >= total {} (rows={:?}, m={m}, k={k})",
            a_tree.num_leaves(),
            a_rows_us
        ));
    }
    // row indices deben estar dentro de la matriz (sanity del pool).
    if let Some(&bad) = a_rows_us.iter().find(|&&r| r >= m) {
        return Err(format!("A: row index {bad} >= m={m}"));
    }
    let a_proof = a_tree.get_multileaf_proof(&a_leaf_idx);

    // B: árbol cacheado por job.
    let b_leaf_idx = MerkleTree::compute_leaf_indices_from_rows(&b_cols_us, (n, k));
    if let Some(&bad) = b_leaf_idx.iter().find(|&&i| i >= b_state.tree.num_leaves()) {
        return Err(format!(
            "B: leaf idx {bad} >= total {} (cols={:?}, n={n}, k={k})",
            b_state.tree.num_leaves(),
            b_cols_us
        ));
    }
    if let Some(&bad) = b_cols_us.iter().find(|&&c| c >= n) {
        return Err(format!("B: col index {bad} >= n={n}"));
    }
    let b_proof = b_state.tree.get_multileaf_proof(&b_leaf_idx);

    Ok(PlainProof {
        m,
        n,
        k,
        noise_rank: r,
        a: MatrixMerkleProof {
            proof: a_proof,
            row_indices: a_rows_us,
        },
        bt: MatrixMerkleProof {
            proof: b_proof,
            row_indices: b_cols_us,
        },
        moe: None, // la softfork densa rechaza MoE
    })
}

/// bincode V1 → gzip → base64 (lo que HeroMiners espera en `plain_proof`).
pub fn encode_share(proof: &PlainProof) -> String {
    let wire = proof.to_wire(); // bincode V1 (SRBMiner-compatible, validado)
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    use std::io::Write as _;
    enc.write_all(&wire).expect("gzip write");
    let gz = enc.finish().expect("gzip finish");
    base64::engine::general_purpose::STANDARD.encode(gz)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LA propiedad de consenso que hace que todo el share funcione:
    /// root(MerkleTree(padded, key)) == blake3::keyed_hash(key, padded).
    /// El tensor_hash de la GPU produce exactamente ese keyed hash.
    #[test]
    fn merkle_root_equals_keyed_blake3_of_padded() {
        for size in [1024usize, 2048, 5000, 65536, 100_000] {
            let key = [0x33u8; 32];
            let data = vec![0x5Au8; size];
            let padded = pad_to_chunk_boundary(&data);
            let tree = MerkleTree::new(&padded, key);
            let root = tree.root();
            let kh = *blake3::keyed_hash(&key, &padded).as_bytes();
            assert_eq!(
                root, kh,
                "raíz del árbol ≠ keyed blake3 para size={size}"
            );
        }
    }

    /// Flujo completo en miniatura: expandir "B", construir árboles, abrir
    /// filas/columnas, y verificar que los proofs reconstruyen las raíces
    /// (la misma verificación de membresía que hace la pool).
    #[test]
    fn share_proof_membership_roundtrip() {
        let (m, n, k) = (4usize, 8usize, 2048usize);
        let key = [0x11u8; 32];
        let a_bytes: Vec<u8> = (0..m * k).map(|i| (i % 127) as i8 as u8).collect();
        let b_bytes: Vec<u8> = (0..n * k).map(|i| ((i * 7) % 127) as i8 as u8).collect();

        let b_state = BState::build(&b_bytes, &key);
        let a_rows = [1u32, 3];
        let b_cols = [0u32, 5, 7];
        let proof = build_share(&key, &a_bytes, &b_state, &a_rows, &b_cols, m, n, k, 128).unwrap();

        assert_eq!(proof.m, m);
        assert_eq!(proof.n, n);
        assert_eq!(proof.k, k);
        assert_eq!(proof.noise_rank, 128);
        assert_eq!(proof.a.row_indices, vec![1, 3]);
        assert_eq!(proof.bt.row_indices, vec![0, 5, 7]);

        // Raíces comprometidas == keyed blake3 de los buffers padeados.
        let a_padded = pad_to_chunk_boundary(&a_bytes);
        let expect_a = *blake3::keyed_hash(&key, &a_padded).as_bytes();
        assert_eq!(proof.a.proof.root, expect_a);
        assert_eq!(proof.bt.proof.root, b_state.tree.root());

        // Membresía: compute_root(key) debe reconstruir la raíz desde las
        // hojas abiertas + siblings (esto es lo que verifica la pool).
        assert_eq!(proof.a.proof.compute_root(key), Some(expect_a));
        assert_eq!(proof.bt.proof.compute_root(key), Some(b_state.tree.root()));

        // Leaf data coherente: cada hoja abierta son 1024 bytes del buffer.
        for (i, &li) in proof.a.proof.leaf_indices.iter().enumerate() {
            let chunk = &a_padded[li * 1024..(li + 1) * 1024];
            assert_eq!(&proof.a.proof.leaf_data[i], chunk);
        }
    }

    #[test]
    fn encode_share_gunzips_to_wire() {
        let (m, n, k) = (2usize, 4usize, 2048usize);
        let key = [0x22u8; 32];
        let a = vec![1u8; m * k];
        let b = vec![2u8; n * k];
        let b_state = BState::build(&b, &key);
        let proof = build_share(&key, &a, &b_state, &[0], &[1], m, n, k, 128).unwrap();
        let b64 = encode_share(&proof);
        assert!(!b64.is_empty());

        // gunzip → debe re-decodificar al mismo PlainProof (wire V1).
        let gz = base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .expect("base64");
        let mut dec = flate2::read::GzDecoder::new(&gz[..]);
        let mut wire = Vec::new();
        use std::io::Read as _;
        dec.read_to_end(&mut wire).expect("gunzip");
        let back = PlainProof::deserialize_compat(&wire).expect("decode wire");
        assert_eq!(back.m, m);
        assert_eq!(back.k, k);
        assert_eq!(back.a.row_indices, vec![0]);
        assert_eq!(back.bt.row_indices, vec![1]);
        assert_eq!(back.a.proof.root, proof.a.proof.root);
    }

    #[test]
    fn boundary_tile_share_builds() {
        let key = [0x11u8; 32];
        let m = 8320usize; let n = 30720usize; let k = 32768usize;
        let a_rows = [8195u32, 8203];
        let mut b_cols: Vec<u32> = Vec::new();
        for j in 0..32u32 { b_cols.push(30464 + j*8); b_cols.push(30464 + j*8 + 1); }
        let a = vec![0u8; m*k];
        let b = vec![0u8; n*k];
        let b_state = BState::build(&b, &key);
        let p = build_share(&key, &a, &b_state, &a_rows, &b_cols, m, n, k, 128).unwrap();
        println!("boundary OK: a_rows={:?} b_cols={} leaves_a={} leaves_b={}",
            p.a.row_indices, p.bt.row_indices.len(),
            p.a.proof.leaf_indices.len(), p.bt.proof.leaf_indices.len());
    }
}

