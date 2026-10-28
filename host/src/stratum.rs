//! Cliente stratum de HeroMiners (JSON-RPC sobre TCP, `type: "v2"`).
//!
//! Validado en vivo contra br.pearl.herominers.com:1200 (probe_stratum.py):
//!   authorize → {"id":1,"error":null,"result":true,"type":"v2"}
//!   notify    → params objeto: {job_id, header(152 hex), target(64 hex BE),
//!                               height, cert_version}

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Job {
    pub job_id: String,
    pub header: [u8; 76],
    pub target_be: [u8; 32],
    pub height: u64,
    pub cert_version: u32,
}

#[derive(Debug)]
pub enum StratumMsg {
    Job(Job),
    Response { id: u64, result: bool, error: Option<String> },
    Disconnected(String),
}

pub struct StratumClient {
    writer: std::sync::Mutex<TcpStream>,
    pub jobs_rx: mpsc::Receiver<Job>,
    pub resp_rx: mpsc::Receiver<(u64, bool, Option<String>)>,
    next_id: AtomicU64,
    worker: Option<std::thread::JoinHandle<()>>,
}

fn hex_to_bytes(hex: &str, out: &mut [u8]) -> Result<(), String> {
    let hex = hex.trim();
    if hex.len() != out.len() * 2 {
        return Err(format!("hex de {} bytes, esperaba {}", hex.len() / 2, out.len()));
    }
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = (chunk[0] as char).to_digit(16).ok_or("hex inválido")?;
        let lo = (chunk[1] as char).to_digit(16).ok_or("hex inválido")?;
        out[i] = (hi * 16 + lo) as u8;
    }
    Ok(())
}

impl StratumClient {
    pub fn connect(host: &str, port: u16) -> Result<Self, String> {
        let addr = format!("{host}:{port}");
        let stream = TcpStream::connect(&addr).map_err(|e| format!("conectar {addr}: {e}"))?;
        stream.set_nodelay(true).ok();
        let writer = stream.try_clone().map_err(|e| format!("clonar socket: {e}"))?;

        let (job_tx, jobs_rx) = mpsc::channel();
        let (resp_tx, resp_rx) = mpsc::channel();
        let reader = BufReader::new(stream);

        let worker = std::thread::Builder::new()
            .name("stratum-reader".into())
            .spawn(move || {
                for line in reader.lines() {
                    match line {
                        Ok(l) => {
                            if l.trim().is_empty() {
                                continue;
                            }
                            match serde_json::from_str::<serde_json::Value>(&l) {
                                Ok(v) => {
                                    let method = v.get("method").and_then(|m| m.as_str());
                                    match method {
                                        Some("mining.notify") => {
                                            if let Some(job) = parse_notify(&v) {
                                                if job_tx.send(job).is_err() {
                                                    return;
                                                }
                                            } else {
                                                eprintln!("[stratum] notify ilegible: {l}");
                                            }
                                        }
                                        Some(other) => {
                                            eprintln!("[stratum] método ignorado: {other}");
                                        }
                                        None => {
                                            // Respuesta a un request nuestro.
                                            let id =
                                                v.get("id").and_then(|i| i.as_u64()).unwrap_or(0);
                                            let result = v
                                                .get("result")
                                                .and_then(|r| r.as_bool())
                                                .unwrap_or(false);
                                            let error = v.get("error").and_then(|e| {
                                                if e.is_null() {
                                                    None
                                                } else {
                                                    Some(e.to_string())
                                                }
                                            });
                                            if resp_tx.send((id, result, error)).is_err() {
                                                return;
                                            }
                                        }
                                    }
                                }
                                Err(e) => eprintln!("[stratum] línea no-JSON: {e}"),
                            }
                        }
                        Err(e) => {
                            let _ = resp_tx.send((0, false, Some(format!("desconectado: {e}"))));
                            return;
                        }
                    }
                }
                let _ = resp_tx.send((0, false, Some("conexión cerrada".into())));
            })
            .map_err(|e| format!("spawn reader: {e}"))?;

        Ok(StratumClient {
            writer: std::sync::Mutex::new(writer),
            jobs_rx,
            resp_rx,
            next_id: AtomicU64::new(1),
            worker: Some(worker),
        })
    }

    fn send(&self, body: &serde_json::Value) -> Result<(), String> {
        let mut s = serde_json::to_string(body).unwrap();
        s.push('\n');
        let mut w = self.writer.lock().map_err(|_| "writer envenenado")?;
        w.write_all(s.as_bytes())
            .map_err(|e| format!("enviar: {e}"))
    }

    fn authorize_body(&self, id: u64, wallet: &str, worker: &str, agent: &str) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "method": "mining.authorize",
            "params": {
                "wallet": wallet,
                "worker": worker,
                "agent": agent,
                "type": "v2",
            }
        })
    }

    pub fn authorize(&self, wallet: &str, worker: &str, agent: &str) -> Result<(), String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = self.authorize_body(id, wallet, worker, agent);
        self.send(&body)?;
        self.wait_response(id, Duration::from_secs(15))
    }

    /// Cambia de wallet sin bloquear (el fee la usa para no perder shares).
    pub fn authorize_async(&self, wallet: &str, worker: &str, agent: &str) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = self.authorize_body(id, wallet, worker, agent);
        if let Err(e) = self.send(&body) {
            eprintln!("[stratum] authorize async falló: {e}");
        }
        id
    }

    /// Envía un share sin bloquear el hilo de minado.
    pub fn submit(&self, job_id: &str, plain_proof_b64: &str) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = serde_json::json!({
            "id": id,
            "method": "mining.submit",
            "params": {
                "job_id": job_id,
                "plain_proof": plain_proof_b64,
            }
        });
        if let Err(e) = self.send(&body) {
            eprintln!("[stratum] submit falló al enviar: {e}");
        }
        id
    }

    pub fn wait_response(
        &self,
        id: u64,
        timeout: Duration,
    ) -> Result<(), String> {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            match self.resp_rx.recv_timeout(deadline - std::time::Instant::now()) {
                Ok((rid, ok, err)) => {
                    if rid == id {
                        if ok {
                            return Ok(());
                        }
                        return Err(err.unwrap_or_else(|| "rechazado".into()));
                    }
                    // Respuesta de otro id: ignorar (submit async).
                }
                Err(mpsc::RecvTimeoutError::Timeout) => return Err("timeout esperando respuesta".into()),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("conexión cerrada".into())
                }
            }
        }
        Err("timeout esperando respuesta".into())
    }

    /// Drena y devuelve el último job pendiente (None si no hay ninguno).
    pub fn latest_job(&self) -> Option<Job> {
        let mut latest = None;
        while let Ok(j) = self.jobs_rx.try_recv() {
            latest = Some(j);
        }
        latest
    }

    pub fn wait_first_job(&self, timeout: Duration) -> Result<Job, String> {
        self.jobs_rx
            .recv_timeout(timeout)
            .map_err(|e| format!("esperando primer job: {e}"))
    }
}

fn parse_notify(v: &serde_json::Value) -> Option<Job> {
    let p = v.get("params")?;
    let job_id = p.get("job_id")?.as_str()?.to_string();
    let header_hex = p.get("header")?.as_str()?;
    let target_hex = p.get("target")?.as_str()?;
    let height = p.get("height").and_then(|h| h.as_u64()).unwrap_or(0);
    let cert_version = p.get("cert_version").and_then(|c| c.as_u64()).unwrap_or(1) as u32;

    let mut header = [0u8; 76];
    hex_to_bytes(header_hex, &mut header).ok()?;
    let mut target_be = [0u8; 32];
    hex_to_bytes(target_hex, &mut target_be).ok()?;

    Some(Job {
        job_id,
        header,
        target_be,
        height,
        cert_version,
    })
}

impl Drop for StratumClient {
    fn drop(&mut self) {
        // Cerrar el socket hace que el reader-thread termine solo.
        if let Ok(w) = self.writer.lock() {
            let _ = w.shutdown(std::net::Shutdown::Both);
        }
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_notify_del_probe_real() {
        // Frame con la forma real capturada de br.pearl.herominers.com:1200
        // (header: version LE 0x20000000 + resto cero → 152 hex).
        let header_hex = format!("00000020{}", "00".repeat(72));
        let frame = format!(
            r#"{{"id": null, "method": "mining.notify", "params": {{"job_id": "00000000_2097152", "header": "{header_hex}", "target": "00000000000007fff80000000000000000000000000000000000000000000000", "height": 132999, "cert_version": 3}}}}"#
        );
        let v: serde_json::Value = serde_json::from_str(&frame).unwrap();
        let job = parse_notify(&v).expect("notify debe parsear");
        assert_eq!(job.job_id, "00000000_2097152");
        assert_eq!(job.header.len(), 76);
        assert_eq!(job.target_be.len(), 32);
        assert_eq!(job.height, 132999);
        assert_eq!(job.cert_version, 3);
        // El header empieza con la version LE 0x20000000 (bytes 00 00 00 20).
        assert_eq!(&job.header[0..4], &[0x00, 0x00, 0x00, 0x20]);
        // Target BE: primeros 6 bytes cero, byte 6 = 0x07, byte 7 = 0xff.
        assert_eq!(&job.target_be[6..8], &[0x07, 0xff]);
    }

    #[test]
    fn hex_roundtrip() {
        let mut out = [0u8; 4];
        hex_to_bytes("deadbeef", &mut out).unwrap();
        assert_eq!(out, [0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(hex_to_bytes("zz", &mut [0u8; 1]).is_err());
        assert!(hex_to_bytes("abc", &mut [0u8; 1]).is_err());
    }
}
