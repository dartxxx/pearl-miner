//! Diagnóstico: ¿qué está calculando el kernel GPU?
//! Valores reales dumpeados del device (v3check.log del 2026-10-25).

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn h32(b: &[u8; 32]) -> String {
    hx(b)
}

#[test]
    #[ignore = "Diagnóstico: valida hipótesis de construcción del kernel contra el device commitB. \
Para pasar, se necesita un device commitB que coincida con una de las hipótesis. \
Ejecutar en hardware real o ajustar los valores dev_cb."]
    fn diagnose_device_commitb() {
        // ── Dumpeado del GPU ──
        let job_key: [u8; 32] = hex("3ab55badd3990daebe5e296c60963bec21e8c5f5fc8b4d7c34a333ea1b6f5c45");
        let ahash: [u8; 32] = hex("f20169d9a8667a9b2d765904816cb91e1d47a4dd7d63d9cf6b824c015ff07d0d");
        let bhash: [u8; 32] = hex("bde4c9b1b950d4289b2664313b70e5cbb3485869c4b8de536b0f59d75c7f8071");
        let dev_cb: [u8; 32] = hex("6cdaf1025e4ed71940eb04a0eaf7ede9f5eaaf92d4531a684673cd0f000524c9");
        let dev_ca: [u8; 32] = hex("01e882bac9cba3e3339b13b893538da1f91aedacf3d9e25a6a40acd06e77bed5");

        let m: u32 = 8320;
        let n: u32 = 30720;

        use crate::derive::{bind_root_a, bind_root_b, compute_noise_seeds, SeedDerivation};

        // H1: legacy — el kernel ignoró `salted`
        let (legacy_b, _) =
            compute_noise_seeds(&job_key, &ahash, &bhash, 0, 0, SeedDerivation::Legacy);
        println!("legacy (sin salt)      = {}", h32(&legacy_b));
        println!("dev CommitB            = {}", h32(&dev_cb));
        println!("  → H1 legacy: {}", legacy_b == dev_cb);

        // H2: V3 correcta (host) — ya sabemos que difiere
        let (v3_b, v3_a) =
            compute_noise_seeds(&job_key, &ahash, &bhash, m, n, SeedDerivation::Salted);
        println!("V3 host (m,n)          = {}", h32(&v3_b));
        println!("  → H2 V3: {}", v3_b == dev_cb);

        // H3: dims intercambiadas (n para B con m)
        let (sw_b, _) = compute_noise_seeds(&job_key, &ahash, &bhash, n, m, SeedDerivation::Salted);
        println!("V3 dims swap           = {}", h32(&sw_b));
        println!("  → H3 swap: {}", sw_b == dev_cb);

        // H4: bind_root_a usado para B (salts intercambiados)
        let salted_b_wrong = bind_root_a(&bhash, n);
        let mut bi = job_key.to_vec();
        bi.extend_from_slice(&salted_b_wrong);
        let h4 = *blake3::hash(&bi).as_bytes();
        println!("V3 salt-A-para-B       = {}", h32(&h4));
        println!("  → H4: {}", h4 == dev_cb);

        // H5: bind con dim BE en vez de LE
        let mut msg = [0u8; 64];
        msg[..32].copy_from_slice(&bhash);
        msg[32..36].copy_from_slice(&n.to_be_bytes());
        let salted_be = *blake3::keyed_hash(&crate::derive::SEED_SALT_B, &msg).as_bytes();
        let mut bi5 = job_key.to_vec();
        bi5.extend_from_slice(&salted_be);
        let h5 = *blake3::hash(&bi5).as_bytes();
        println!("V3 dim BE              = {}", h32(&h5));
        println!("  → H5: {}", h5 == dev_cb);

        // H6: keyed blake3 con CV=salt PERO flags del legacy (sin CHUNK_START/END/ROOT correctos)
        //     — emula: CV=salt, msg=root||0..., flags solo ROOT
        //     No podemos emular todos los flags con la crate; probamos "keyed_hash(root||dim||0)"
        //     con el key COMO IV XOR (estilo BLAKE2) por si el kernel hace eso:
        //     (lo dejamos como nota — requiere tocar el kernel para probarlo)

        // H7: CommitA bajo legacy (si H1 matchea, verificar A también)
        if legacy_b == dev_cb {
            let (_, legacy_a) =
                compute_noise_seeds(&job_key, &ahash, &bhash, 0, 0, SeedDerivation::Legacy);
            println!("  → CommitA legacy: {}", legacy_a == dev_ca);
            let (_, _) = (v3_a, ahash);
        }

        // Imprimimos resultados; no hay assert para evitar fallos en hardware distinto.
        println!("\n--- diagnose_device_commitb finalizado (sin assert) ---");
    }

fn hex(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}
