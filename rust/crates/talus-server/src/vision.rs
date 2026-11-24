//! Option-B vision glue: drives Strata's `strata-vision` image encoder as a subprocess and hands
//! the embeddings to the engine as GENI's SVE1 file, mirroring `Vision` in Strata's `serve/server.py`.
//!
//! Encoder protocol (verified against `tools/vision/own/own_reader.cpp`):
//!   encoder -> `READY <out_dim> own` once the mmproj is loaded
//!   us      -> `ENC <image-file> <out-file>` (the last space splits the two paths, so the out
//!             path may not hold one; ours are hex names)
//!   encoder -> `OK <n> <nx> <ny> <ms>` and wrote an SVE1 record to `<out-file>`, or `ERR <reason>`
//!   us      -> `QUIT` to stop
//! An `ERR` line leaves the encoder alive (it is one picture's refusal). A vanished line stream
//! means the process died: the picture is retried once against a fresh spawn.
//!
//! Caching is the server's job, not the encoder's (`own_reader` re-encodes every `ENC`): each
//! picture's embeddings are content-addressed as `<dir>/<key>.sve`, where `key` is the first 8
//! bytes of the picture bytes' SHA-256 in hex. That hex is also the GENI `img_keys=` value
//! (#2228), so the engine's conversation cache names the picture the same way. A request's
//! images are concatenated into one combine file named for the SHA-256 of the concatenation:
//! keyed on content, never on size, so two different pictures of equal bytes never share a
//! file (the same-size collision that voided PR #7's vision eval).

use crate::json::JsonValue;
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::time::Duration;

// ---------------------------------------------------------------- sha256 (hand-rolled: the crate keeps its no-dependency rule)

const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
const SHA_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 over `data`. Verified against `sha2(1)` output vectors in the tests below.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = SHA_H0;
    let bits = (data.len() as u64) * 8;
    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for j in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA_K[j])
                .wrapping_add(w[j]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let mj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(mj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(v);
        }
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn sha256_hex(data: &[u8]) -> String {
    sha256(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// The picture's cache key and GENI `img_keys=` value (#2228): SHA-256's first 8 bytes in hex.
/// The engine parses these hex values as u64 keys, so exactly 16 hex digits, no more.
pub fn picture_key(bytes: &[u8]) -> String {
    let h = sha256(bytes);
    h[..8].iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------- base64 (standard and URL-safe alphabets, padding optional)

/// Decode base64 with either alphabet; `None` on malformed input. Tolerates the absent padding
/// data-URL payloads sometimes carry, and embedded newlines (some clients wrap at 76 columns).
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u64> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u64),
            b'a'..=b'z' => Some((c - b'a' + 26) as u64),
            b'0'..=b'9' => Some((c - b'0' + 52) as u64),
            b'+' | b'-' => Some(62),
            b'/' | b'_' => Some(63),
            _ => None,
        }
    }
    let mut acc: u64 = 0;
    let mut bits = 0u32;
    let mut out = Vec::with_capacity(s.len() * 3 / 4 + 1);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        i += 1;
        if c == b'=' {
            // padding: only ever at the end, and nothing after it but the other '='
            if bytes[i..]
                .iter()
                .any(|&b| b != b'=' && b != b'\r' && b != b'\n')
            {
                return None;
            }
            break;
        }
        if c == b'\r' || c == b'\n' || c == b' ' {
            continue;
        }
        let v = val(c)?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    if bits >= 6 {
        return None; // a full byte minus one 6-bit group: a leftover sextet is malformed
    }
    Some(out)
}

/// A data URL's payload (`data:image/png;base64,<b64>`); `None` when it is not one. A URL-safe
/// payload decodes the same way (`val()` accepts both alphabets).
pub fn decode_data_url(url: &str) -> Option<Vec<u8>> {
    let rest = url.get(5..).filter(|_| url.starts_with("data:"))?;
    let (_, b64) = rest.split_once(',')?;
    base64_decode(b64)
}

/// Fetch an image's bytes from `http(s)://`, `file://`, a plain path, or a data URL.
/// HTTP goes through `curl` (the crate has no HTTP client and keeps it that way, as `server.py` did).
pub fn fetch_bytes(source: &str, timeout_secs: u64) -> Result<Vec<u8>, String> {
    if let Some(b) = decode_data_url(source) {
        return Ok(b);
    }
    if let Some(p) = source.strip_prefix("file://") {
        return std::fs::read(p).map_err(|e| format!("vision: cannot read {p}: {e}"));
    }
    let is_url = source.starts_with("http://") || source.starts_with("https://");
    if !is_url {
        return std::fs::read(source).map_err(|e| format!("vision: cannot read {source}: {e}"));
    }
    let out = Command::new("curl")
        .args(["-sS", "-L", "--max-time"])
        .arg(timeout_secs.to_string())
        .arg(source)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("vision: fetching {source} needs curl on PATH ({e})"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!(
            "vision: curl failed for {source} ({}: {})",
            out.status.code().unwrap_or(-1),
            if err.is_empty() {
                "no error text"
            } else {
                &err
            }
        ));
    }
    if out.stdout.is_empty() {
        return Err(format!("vision: {source} returned no bytes"));
    }
    Ok(out.stdout)
}

// ---------------------------------------------------------------- format sniffing (the encoder's image reader is stb: no WebP)

/// The formats `image_load_file` (stb) actually reads, by magic. `Err` carries the refusal's text:
/// WebP gets its own, since clients send it often and "unsupported image format" would not say why.
pub fn sniff(bytes: &[u8]) -> Result<&'static str, String> {
    let head = [0u8; 12];
    let h = if bytes.len() >= 12 {
        let mut b = head;
        b.copy_from_slice(&bytes[..12]);
        b
    } else {
        let mut b = head;
        b[..bytes.len()].copy_from_slice(bytes);
        b
    };
    if h[..3] == [0xff, 0xd8, 0xff] {
        return Ok("jpeg");
    }
    if h[..4] == [0x89, b'P', b'N', b'G'] {
        return Ok("png");
    }
    if h[..2] == *b"BM" {
        return Ok("bmp");
    }
    if h[..4] == *b"GIF8" {
        return Ok("gif");
    }
    if h[..4] == *b"RIFF" && h[8..12] == *b"WEBP" {
        return Err("vision: WebP is not readable by the image encoder \
                    (it handles JPEG, PNG, BMP and GIF): convert the picture first"
            .to_string());
    }
    Err(format!(
        "vision: unrecognised image format (first bytes {:02x}{:02x}): the encoder reads JPEG, PNG, BMP and GIF",
        h[0], h[1]
    ))
}

// ---------------------------------------------------------------- run-config block

/// The run config's `"vision"` block: the encoder subprocess and its cache.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    /// the `strata-vision` (or `strata-vision-hip`) binary
    pub exe: PathBuf,
    /// the model's mmproj file: vision tower + projector, its `out_dim` must equal the model's `n_embd`
    pub mmproj: PathBuf,
    /// accepted by the CLI and ignored (our tower never tokenizes the text model) — kept so
    /// `server.py` run configs transfer unchanged
    pub model: Option<PathBuf>,
    /// GPU encode (`--gpu`); default false, matching `server.py`, so the encoder never contends
    /// with the engine's GPU work
    pub gpu: bool,
    pub threads: Option<usize>,
    pub max_image_tokens: Option<usize>,
    pub min_image_tokens: Option<usize>,
    /// how long one `ENC` may take before the encoder is declared dead mid-picture
    pub enc_timeout_secs: u64,
    /// how long `READY` may take after a spawn (mmproj load + GPU warm-up)
    pub ready_timeout_secs: u64,
    /// `LD_LIBRARY_PATH` entries for the encoder child (its HIP runtime may live outside the
    /// default search path, as for the engine's `lib_dirs`)
    pub lib_dirs: Vec<String>,
    /// where the `.img`/`.sve` files live; default `<temp>/talus-vision-<pid>`
    pub cache_dir: Option<PathBuf>,
    /// cached pictures before the least-recently-used one is dropped (and its `.sve` deleted)
    pub cache_entries: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            exe: PathBuf::new(),
            mmproj: PathBuf::new(),
            model: None,
            gpu: false,
            threads: None,
            max_image_tokens: None,
            min_image_tokens: None,
            enc_timeout_secs: 120,
            ready_timeout_secs: 300,
            lib_dirs: Vec::new(),
            cache_dir: None,
            cache_entries: 64,
        }
    }
}

impl VisionConfig {
    /// Parse the `"vision"` block. `exe` and `mmproj` are required non-empty strings; paths are
    /// taken as written (the engine's args are, so the run config's culture is absolute paths).
    pub fn from_json(v: &JsonValue) -> Result<Self, String> {
        let mut c = Self::default();
        if v.as_object().is_none() {
            return Err("vision: the \"vision\" config key must be an object".to_string());
        }
        let req = |k: &str| -> Result<PathBuf, String> {
            v.get_str(k)
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
                .ok_or_else(|| {
                    format!("vision: \"{k}\" is required (path to the encoder binary / mmproj)")
                })
        };
        c.exe = req("exe")?;
        c.mmproj = req("mmproj")?;
        c.model = v
            .get_str("model")
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from);
        if let Some(g) = v.get_bool("gpu") {
            c.gpu = g;
        }
        if let Some(n) = v.get("max_tokens").and_then(|x| x.as_i64()) {
            c.max_image_tokens = Some(n.max(1) as usize);
        }
        if let Some(n) = v.get("min_tokens").and_then(|x| x.as_i64()) {
            c.min_image_tokens = Some(n.max(1) as usize);
        }
        if let Some(n) = v.get("threads").and_then(|x| x.as_i64()) {
            c.threads = Some(n.max(1) as usize);
        }
        if let Some(n) = v.get("enc_timeout_secs").and_then(|x| x.as_i64()) {
            c.enc_timeout_secs = n.max(1) as u64;
        }
        if let Some(n) = v.get("ready_timeout_secs").and_then(|x| x.as_i64()) {
            c.ready_timeout_secs = n.max(1) as u64;
        }
        if let Some(dir) = v.get_str("cache_dir").filter(|s| !s.trim().is_empty()) {
            c.cache_dir = Some(PathBuf::from(dir));
        }
        if let Some(n) = v.get("cache_entries").and_then(|x| x.as_i64()) {
            c.cache_entries = n.max(1) as usize;
        }
        if let Some(arr) = v.get("lib_dirs").and_then(|x| x.as_array()) {
            c.lib_dirs = arr
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect();
        }
        Ok(c)
    }
}

// ---------------------------------------------------------------- encoder subprocess

struct EncoderProc {
    child: Child,
    stdin: std::process::ChildStdin,
    lines: Receiver<String>,
}

impl EncoderProc {
    fn spawn(cfg: &VisionConfig) -> Result<Self, String> {
        let mut cmd = Command::new(&cfg.exe);
        cmd.arg("--mmproj").arg(&cfg.mmproj);
        if cfg.gpu {
            cmd.arg("--gpu");
        }
        if let Some(t) = cfg.threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        if let Some(m) = cfg.max_image_tokens {
            cmd.arg("--max-tokens").arg(m.to_string());
        }
        if let Some(m) = cfg.min_image_tokens {
            cmd.arg("--min-tokens").arg(m.to_string());
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if !cfg.lib_dirs.is_empty() {
            let joined = cfg.lib_dirs.join(":");
            let prior = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let value = if prior.is_empty() {
                joined
            } else {
                format!("{joined}:{prior}")
            };
            cmd.env("LD_LIBRARY_PATH", value);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("vision: cannot start {}: {e}", cfg.exe.display()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "vision: the encoder has no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "vision: the encoder has no stdout".to_string())?;
        let (tx, rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break, // EOF or a read error: the process is gone
                    Ok(_) => {
                        let trimmed = line.trim_end().to_string();
                        if !trimmed.is_empty() && tx.send(trimmed).is_err() {
                            break; // nobody listening any more
                        }
                    }
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            lines: rx,
        })
    }

    /// One reply line, or `Err(())` on timeout/EOF (the caller treats that as death).
    fn reply(&mut self, timeout: Duration) -> Result<String, ()> {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => Ok(line),
            Err(RecvTimeoutError::Timeout) => Err(()),
            Err(RecvTimeoutError::Disconnected) => Err(()),
        }
    }

    /// `QUIT` and reap; a lost QUIT (dead already, stdin closed) still reaps.
    fn shutdown(&mut self) {
        let _ = writeln!(self.stdin, "QUIT");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for EncoderProc {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// One encoded picture: the `.sve` file holding its single SVE1 record, and the record's grid.
#[derive(Clone, Debug)]
pub struct Cached {
    pub key: String,
    pub path: PathBuf,
    pub n: i64,
    pub nx: i64,
    pub ny: i64,
}

/// What a frontend needs to drive one GENI line: the engine's embeddings file, the `img_keys=`
/// value, and per-picture grids (for splicing exactly `n` image-pad tokens per picture).
#[derive(Debug)]
pub struct GeniPayload {
    pub combine: PathBuf,
    pub img_keys: String,
    pub pictures: Vec<Cached>,
}

enum EncErr {
    /// the encoder answered: this picture is refused, the process is fine
    Refused(String),
    /// no answer (timeout/EOF): the process is untrustworthy, respawn and retry once
    Dead(String),
}

struct State {
    live: Option<EncoderProc>,
    cache: HashMap<String, Cached>,
    /// LRU: front = most recently used; eviction deletes the `.sve`
    order: VecDeque<String>,
    /// the encoder's SVE width (`READY <out_dim>`), known after the first spawn
    out_dim: Option<i64>,
}

impl State {
    fn touch(&mut self, key: &str) {
        self.order.retain(|k| k != key);
        self.order.push_front(key.to_string());
    }

    fn evict(&mut self, cap: usize) {
        while self.order.len() > cap {
            let Some(old) = self.order.pop_back() else {
                break;
            };
            if let Some(c) = self.cache.remove(&old) {
                let _ = std::fs::remove_file(c.path);
            }
        }
    }
}

/// The server's handle on the encoder: one process, serialized under a single lock (which the
/// one-pipe protocol requires anyway), content-addressed picture cache in front of it.
pub struct VisionEncoder {
    cfg: VisionConfig,
    dir: PathBuf,
    state: Mutex<State>,
}

impl VisionEncoder {
    pub fn new(cfg: VisionConfig) -> Result<Self, String> {
        let dir = cfg.cache_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("talus-vision-{}", std::process::id()))
        });
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("vision: cannot create cache dir {}: {e}", dir.display()))?;
        Ok(Self {
            cfg,
            dir,
            state: Mutex::new(State {
                live: None,
                cache: HashMap::new(),
                order: VecDeque::new(),
                out_dim: None,
            }),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The encoder's embedding width once spawned (the engine verifies it equals `n_embd` when it
    /// reads the records; this lets the frontend warn early).
    pub fn out_dim(&self) -> Option<i64> {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).out_dim
    }

    /// Encode a whole request's pictures (in prompt order) and build its combine file under one
    /// lock, so no eviction can delete a part between encoding and concatenation.
    pub fn encode_request(&self, images: &[Vec<u8>]) -> Result<GeniPayload, String> {
        if images.is_empty() {
            return Err("vision: a picture request with no pictures".to_string());
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut pictures = Vec::with_capacity(images.len());
        for bytes in images {
            pictures.push(self.encode_locked(&mut st, bytes)?);
        }
        let combine = self.combine_locked(&pictures)?;
        Ok(GeniPayload {
            img_keys: pictures
                .iter()
                .map(|p| p.key.as_str())
                .collect::<Vec<_>>()
                .join(","),
            combine,
            pictures,
        })
    }

    /// One picture: cache hit, or an `ENC` against the (respawned-on-death) encoder.
    fn encode_locked(&self, st: &mut State, bytes: &[u8]) -> Result<Cached, String> {
        let key = picture_key(bytes);
        if st.cache.contains_key(&key) {
            st.touch(&key);
            return Ok(st.cache[&key].clone());
        }
        let img = self.dir.join(format!("{key}.img"));
        let sve = self.dir.join(format!("{key}.sve"));
        std::fs::write(&img, bytes).map_err(|e| {
            format!(
                "vision: cannot write the picture file {}: {e}",
                img.display()
            )
        })?;
        let mut result = self.enc_once(st, &img, &sve, &key);
        if let Err(EncErr::Dead(first)) = result {
            // respawn once: an encoder that died on this line is given a clean process
            if let Some(mut dead) = st.live.take() {
                let _ = dead.child.kill();
                let _ = dead.child.wait();
            }
            result = match self.enc_once(st, &img, &sve, &key) {
                Err(EncErr::Dead(second)) => Err(EncErr::Refused(format!(
                    "vision: the encoder died encoding a picture twice ({first}; then {second})"
                ))),
                other => other,
            };
        }
        let _ = std::fs::remove_file(&img);
        let (n, nx, ny) = match result {
            Ok(v) => v,
            Err(EncErr::Refused(m) | EncErr::Dead(m)) => return Err(m),
        };
        let cached = Cached {
            key: key.clone(),
            path: sve,
            n,
            nx,
            ny,
        };
        st.cache.insert(key.clone(), cached.clone());
        st.touch(&key);
        st.evict(self.cfg.cache_entries);
        Ok(cached)
    }

    /// One `ENC`/reply exchange; `Err(Dead)` marks the process for respawn.
    fn enc_once(
        &self,
        st: &mut State,
        img: &Path,
        sve: &Path,
        key: &str,
    ) -> Result<(i64, i64, i64), EncErr> {
        if st.live.is_none() {
            let mut proc = EncoderProc::spawn(&self.cfg).map_err(EncErr::Refused)?;
            let line = proc
                .reply(Duration::from_secs(self.cfg.ready_timeout_secs))
                .map_err(|_| {
                    EncErr::Dead(format!(
                        "vision: the encoder gave no READY within {} s ({}? mmproj load)",
                        self.cfg.ready_timeout_secs,
                        self.cfg.mmproj.display()
                    ))
                })?;
            match line.split_whitespace().next() {
                Some("READY") => {
                    st.out_dim = line.split_whitespace().nth(1).and_then(|w| w.parse().ok());
                }
                Some("ERR") => {
                    return Err(EncErr::Refused(format!(
                        "vision: the encoder could not start: {}",
                        line.trim_start_matches("ERR").trim()
                    )))
                }
                _ => {
                    return Err(EncErr::Refused(format!(
                        "vision: the encoder's first line was not READY ({line})"
                    )))
                }
            }
            st.live = Some(proc);
        }
        let proc = st.live.as_mut().expect("live set above");
        // the protocol splits at the LAST space: our out path is a hex basename, clean either way
        let cmd = format!("ENC {} {}", img.display(), sve.display());
        if writeln!(proc.stdin, "{cmd}")
            .and_then(|_| proc.stdin.flush())
            .is_err()
        {
            return Err(EncErr::Dead(
                "vision: the encoder's stdin is closed".to_string(),
            ));
        }
        let line = match proc.reply(Duration::from_secs(self.cfg.enc_timeout_secs)) {
            Ok(l) => l,
            Err(()) => {
                let why = match proc.child.try_wait() {
                    Ok(Some(code)) => format!("exited {code}"),
                    Ok(None) => format!("silent for {} s", self.cfg.enc_timeout_secs),
                    Err(_) => "unreachable".to_string(),
                };
                return Err(EncErr::Dead(format!(
                    "vision: the encoder {why} while encoding picture {key}"
                )));
            }
        };
        let mut it = line.split_whitespace();
        match it.next() {
            Some("OK") => {
                let (ns, nxs, nys) = (it.next(), it.next(), it.next());
                let tri = (
                    ns.and_then(|s| s.parse::<i64>().ok()),
                    nxs.and_then(|s| s.parse::<i64>().ok()),
                    nys.and_then(|s| s.parse::<i64>().ok()),
                );
                let (n, nx, ny) = match tri {
                    (Some(n), Some(nx), Some(ny)) => (n, nx, ny),
                    _ => {
                        return Err(EncErr::Refused(format!(
                            "vision: malformed OK from the encoder ({line})"
                        )))
                    }
                };
                if n < 1 || nx < 1 || ny < 1 || nx * ny != n {
                    return Err(EncErr::Refused(format!(
                        "vision: the encoder reported an impossible grid {nx}x{ny}={n}"
                    )));
                }
                Ok((n, nx, ny))
            }
            Some("ERR") => Err(EncErr::Refused(format!(
                "vision: {}",
                line.trim_start_matches("ERR").trim()
            ))),
            _ => Err(EncErr::Refused(format!(
                "vision: unexpected encoder reply ({line})"
            ))),
        }
    }

    /// Concatenate the pictures' SVE records into one file named for their content hash. Written
    /// only if absent; never deleted (the engine opens it later, after the caller's lock is long
    /// gone). Callers that must keep parts from being evicted hold the state lock across this.
    fn combine_locked(&self, pictures: &[Cached]) -> Result<PathBuf, String> {
        let mut buf: Vec<u8> = Vec::new();
        for p in pictures {
            let mut f = std::fs::File::open(&p.path).map_err(|e| {
                format!(
                    "vision: picture {} vanished from the cache before its request was built ({e})",
                    p.key
                )
            })?;
            f.read_to_end(&mut buf)
                .map_err(|e| format!("vision: cannot read {}: {e}", p.path.display()))?;
        }
        let path = self.dir.join(format!(
            "com-{}.sve",
            sha256_hex(&buf).chars().take(32).collect::<String>()
        ));
        if !path.exists() {
            std::fs::write(&path, &buf).map_err(|e| {
                format!(
                    "vision: cannot write the combine file {}: {e}",
                    path.display()
                )
            })?;
        }
        Ok(path)
    }

    /// The GENI embeddings argument for a picture list that was already encoded (used by tests
    /// and, later, by the frontend's retry path).
    pub fn combine_of(&self, pictures: &[Cached]) -> Result<PathBuf, String> {
        let _guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.combine_locked(pictures)
    }
}

impl Drop for VisionEncoder {
    fn drop(&mut self) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut proc) = st.live.take() {
            proc.shutdown();
        }
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // crosses several block boundaries
        assert_eq!(
            sha256_hex(&[b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
        // the padding edge: 55 + 0x80 + 8 length bytes fill exactly one block
        assert_eq!(
            sha256_hex(&[0x61u8; 55]),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
        );
    }

    #[test]
    fn picture_key_is_engine_shaped() {
        let k = picture_key(b"a");
        assert_eq!(k.len(), 16);
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(k, picture_key(b"a"));
        assert_ne!(k, picture_key(b"b"));
        // the engine reads it as strtoull base 16: no prefix, fits a u64
        assert!(u64::from_str_radix(&k, 16).is_ok());
    }

    #[test]
    fn base64_basic() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello"); // no padding
        assert_eq!(base64_decode("").unwrap(), b"");
        assert_eq!(base64_decode("-_Tg==").unwrap(), vec![0xfb, 0xf4, 0xe0]); // URL-safe alphabet
        assert_eq!(base64_decode("aGVsbG8=\n").unwrap(), b"hello"); // wrapped payload
        assert_eq!(base64_decode("///="), Some(vec![0xff, 0xff]));
        assert_eq!(base64_decode("a"), None); // one leftover sextet
        assert_eq!(base64_decode("aGVsbG8=x"), None); // junk after padding
        assert_eq!(base64_decode("??"), None);
    }

    #[test]
    fn data_url_shape() {
        let d = decode_data_url("data:image/png;base64,aGVsbG8=").unwrap();
        assert_eq!(d, b"hello");
        assert!(decode_data_url("https://x/y").is_none());
        assert!(decode_data_url("data:image/png;base64,??!").is_none());
    }

    #[test]
    fn sniff_recognises_supported_and_webp() {
        assert_eq!(sniff(&[0xff, 0xd8, 0xff, 0xe0]).unwrap(), "jpeg");
        assert_eq!(sniff(&[0x89, b'P', b'N', b'G', 0, 0]).unwrap(), "png");
        assert_eq!(sniff(&[b'B', b'M', 1, 2]).unwrap(), "bmp");
        assert_eq!(sniff(b"GIF89a123456").unwrap(), "gif");
        let mut webp = *b"RIFF\x00\x00\x00\x00WEBPVP8 ";
        webp[4..8].copy_from_slice(&42u32.to_le_bytes());
        let e = sniff(&webp).unwrap_err();
        assert!(e.contains("WebP"), "{e}");
        let e = sniff(b"PDF-1.4xxxx").unwrap_err();
        assert!(e.contains("unrecognised"), "{e}");
    }

    #[test]
    fn combine_key_is_content_not_size() {
        // the PR #7 collision, as a regression test: two records of identical length, different
        // content, must not name the same combine file
        let a = vec![0u8; 512];
        let mut b = vec![0u8; 512];
        b[7] = 1;
        let na = sha256_hex(&a);
        let nb = sha256_hex(&b);
        assert_eq!(a.len(), b.len());
        assert_ne!(na, nb);
    }

    #[test]
    fn vision_config_parses_and_demands_paths() {
        let ok = JsonValue::parse(
            r#"{"exe":"/x/strata-vision-hip","mmproj":"/m/mmproj.gguf","gpu":true,
                "max_tokens":1024,"lib_dirs":["/opt/hip"]}"#,
        )
        .unwrap();
        let c = VisionConfig::from_json(&ok).unwrap();
        assert_eq!(c.exe, PathBuf::from("/x/strata-vision-hip"));
        assert!(c.gpu);
        assert_eq!(c.max_image_tokens, Some(1024));
        assert_eq!(c.lib_dirs, vec!["/opt/hip".to_string()]);
        assert_eq!(c.cache_entries, 64);
        let missing = JsonValue::parse(r#"{"mmproj":"/m/x"}"#).unwrap();
        let e = VisionConfig::from_json(&missing).unwrap_err();
        assert!(e.contains("exe"), "{e}");
        let scalar = JsonValue::parse("7").unwrap();
        assert!(VisionConfig::from_json(&scalar).is_err());
    }

    #[test]
    fn encode_against_a_fake_encoder() {
        // a python script speaks READY/OK/QUIT and writes a real SVE1 record: the actual mmproj is
        // not needed to exercise the protocol, the cache, and the respawn-on-death path
        let dir = std::env::temp_dir().join(format!("talus-vision-test-{}", std::process::id()));
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("fake.py");
        let die_once = dir.join("die-once");
        std::fs::write(
            &fake,
            // Rust-source column 0: Python indentation is literal here (the fake runs unmodified)
            r#"#!/usr/bin/env python3
import os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIE = os.path.join(HERE, "die-once")

def out(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

out("READY 2560 own")
for line in sys.stdin:
    line = line.strip()
    if line == "QUIT":
        break
    if not line.startswith("ENC "):
        out("ERR confused")
        continue
    dst = line.split(" ")[-1]
    if os.path.exists(DIE):
        os.remove(DIE)
        sys.exit(9)
    with open(dst, "wb") as f:
        f.write(struct.pack("<5i", 0x31455653, 4, 2, 2, 2560))
        f.write(b"\x00" * (4 * 2560 * 4))
    out("OK 4 2 2 1")
"#,
        )
        .unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let cfg = VisionConfig {
            exe: fake.clone(),
            mmproj: PathBuf::from("/no/mmproj"),
            cache_dir: Some(dir.clone()),
            enc_timeout_secs: 20,
            ..Default::default()
        };
        let enc = VisionEncoder::new(cfg).unwrap();
        let pic = vec![7u8; 40];
        let key1 = picture_key(&pic);
        let payload = enc.encode_request(std::slice::from_ref(&pic)).unwrap();
        assert_eq!(payload.pictures[0].n, 4);
        assert_eq!(payload.img_keys, key1);
        assert!(payload.combine.exists());
        // cache hit: the same bytes reuse the same record file without a second ENC
        let again = enc.encode_request(std::slice::from_ref(&pic)).unwrap();
        assert_eq!(again.pictures[0].path, payload.pictures[0].path);
        assert_eq!(enc.out_dim(), Some(2560));
        // death mid-picture: the encoder exits(9) at the ENC (after its READY), and the retry
        // against a fresh spawn must succeed
        std::fs::write(&die_once, "x").unwrap();
        let third = enc.encode_request(&[vec![9u8; 31]]).unwrap();
        assert_eq!(third.pictures[0].n, 4);
        assert!(!die_once.exists());
        drop(enc);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
