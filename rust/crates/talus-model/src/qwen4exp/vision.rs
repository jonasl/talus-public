//! D3b-2: in-process image encoders for `--vision` (the interface frozen for the native encoder).
//!
//! A GENI request may carry its pictures as pixels instead of an SVE1 file:
//! `GENI <max_new> [keys] img_keys=<hex>,... img_px=<w>x<h>,... @px <ids>`, followed at once by a binary frame (the images'
//! RGB u8, row-major, back to back, exactly sum(w*h*3) bytes, no header). The engine hands the pixels to an [`ImageEncoder`],
//! which writes the embedding rows into device memory it owns; the prompt path copies them into `emb` per chunk (D3b-1's device
//! runs). talus-model does not depend on any encoder crate: talus-vision implements the trait, talus-cli wires it.

use core::ffi::c_void;

/// talus_hip::rt raw stream / event (backend-neutral `void*`).
pub type RawStream = *mut c_void;
pub type RawEvent = *mut c_void;

/// One picture of a request: its resized RGB u8 pixels and the server's key for it (`img_keys`, #2228).
#[derive(Clone, Copy, Debug)]
pub struct PixelImage<'a> {
    pub w: u32,
    pub h: u32,
    pub rgb: &'a [u8],
    pub key: u64,
}

/// One raw picture of an `@raw` request: the encoded bytes (JPEG, PNG, BMP, GIF - the tower's
/// own decoder reads them) and the server's key.  `predict` is the server's prediction of the
/// smart_resize'd size, sent as `img_px` and checked by the encoder against what the tower
/// actually prepared: a Python predictor drifting from the tower's rules is an error, not a
/// silently wrong picture.
#[derive(Clone, Copy, Debug)]
pub struct RawImage<'a> {
    pub data: &'a [u8],
    pub key: u64,
    pub predict: Option<(u32, u32)>,
}

/// One encoded picture: `n` = `nx * ny` rows of `n_embd` f32 in device memory, in merge order, valid until
/// [`ImageEncoder::release`].
#[derive(Clone, Copy, Debug)]
pub struct EncodedImage {
    pub rows: *const f32,
    pub n: i64,
    pub nx: i64,
    pub ny: i64,
}

pub trait ImageEncoder: Send {
    /// checked == the model's n_embd at load
    fn n_embd(&self) -> i64;
    /// rows per request; the engine refuses a request above it (ERR)
    fn max_rows(&self) -> i64;
    /// Enqueue the work on `stream`; returns rows valid until `release()`; records `done` after the last write.
    fn encode(
        &mut self,
        imgs: &[PixelImage],
        stream: RawStream,
        done: RawEvent,
    ) -> Result<Vec<EncodedImage>, String>;
    /// The `@raw` form: encode whole pictures - the encoder decodes, resizes and normalises them
    /// itself (the tower's own `prepare_image`), so the server needed no size agreement beyond
    /// its pad-token count.  An encoder without a tower refuses: the mock only knows resized
    /// pixels.
    fn encode_raw(
        &mut self,
        _raws: &[RawImage],
        _stream: RawStream,
        _done: RawEvent,
    ) -> Result<Vec<EncodedImage>, String> {
        Err("this encoder takes resized pixels only (@px); @raw needs the native tower".into())
    }
    /// `VGEOM`: geometry constants a server's own smart_resize predictor must take from the
    /// engine (align, min_tokens, max_tokens) rather than guess.  Only the tower knows its own
    /// bounds; the mock has none.
    fn geom(&self) -> Option<(i64, i64, i64)> {
        None
    }
    /// called by the engine after the request's prompt pass (also after an ERR or a cancelled read)
    fn release(&mut self);
}

/// Largest frame the engine accepts (the server's cap 4096 bound: 2048 x 2048 x 3).
pub const MAX_FRAME_BYTES: usize = 12 << 20;

/// The `img_px=<w>x<h>,...` list of a GENI line, and whether the line is the pixel form (`@px` where the SVE path stands).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PixelSpec {
    pub sizes: Vec<(u32, u32)>,
    /// sum(w * h * 3), the frame's exact length
    pub bytes: usize,
}

/// `img_px` grammar: `<w>x<h>` entries joined by ',', w and h decimal 1..=65535. None = malformed.
pub fn parse_img_px(v: &str) -> Option<PixelSpec> {
    let mut sizes = Vec::new();
    let mut bytes = 0usize;
    for e in v.split(',') {
        let (w, h) = e.split_once('x')?;
        if w.is_empty()
            || h.is_empty()
            || !w.bytes().all(|b| b.is_ascii_digit())
            || !h.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let (w, h): (u32, u32) = (w.parse().ok()?, h.parse().ok()?);
        if w == 0 || h == 0 || w > 65535 || h > 65535 {
            return None;
        }
        bytes = bytes.checked_add((w as usize).checked_mul(h as usize)?.checked_mul(3)?)?;
        sizes.push((w, h));
    }
    (!sizes.is_empty()).then_some(PixelSpec { sizes, bytes })
}

/// `img_bytes` grammar (the `@raw` form): the frame's exact per-picture byte lengths, `<n>[,<n>...]`
/// decimal, non-empty and capped so a frame stays sane. None = malformed.
pub fn parse_img_bytes(v: &str) -> Option<Vec<u64>> {
    let mut lens = Vec::new();
    for e in v.split(',') {
        if e.is_empty() || !e.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let n: u64 = e.parse().ok()?;
        if n == 0 || n > MAX_FRAME_BYTES as u64 {
            return None;
        }
        lens.push(n);
    }
    (!lens.is_empty()).then_some(lens)
}

/// What the stdin reader must do after a protocol line: `Some(n)` = read exactly `n` raw bytes (the frame) before the next line.
/// Only a GENI line in the pixel form (an `@px` token with its `img_px=` key, or the `@raw` form with
/// `img_bytes=`) carries a frame; a malformed size list = no frame can be framed (the engine
/// answers ERR and the client's bytes then arrive as lines: documented, the server never sends that).
pub fn frame_len(line: &str) -> Option<usize> {
    if !line.starts_with("GENI ") {
        return None;
    }
    let mut px = None;
    let mut by = None;
    let mut at_px = false;
    let mut at_raw = false;
    for tok in line.split(' ') {
        if let Some(v) = tok.strip_prefix("img_px=") {
            px = Some(v);
        } else if let Some(v) = tok.strip_prefix("img_bytes=") {
            by = Some(v);
        } else if tok == "@px" {
            at_px = true;
            break; // ids follow
        } else if tok == "@raw" {
            at_raw = true;
            break; // ids follow
        } else if !tok.contains('=') && tok != "GENI" && !tok.bytes().all(|b| b.is_ascii_digit()) {
            break; // an SVE path: no frame
        }
    }
    if at_raw {
        return by
            .and_then(parse_img_bytes)
            .map(|l| l.iter().sum::<u64>() as usize);
    }
    if !at_px {
        return None;
    }
    px.and_then(parse_img_px).map(|s| s.bytes)
}

/// The stdin reader's frame step: after `line`, read its frame (if it carries one) from `r`. Ok(None) = no frame; Ok(Some(v)) =
/// the frame (empty when over [`MAX_FRAME_BYTES`]: those bytes are read and dropped); Err = the input ended inside the frame.
#[allow(clippy::result_unit_err)]
pub fn read_frame(line: &str, r: &mut impl std::io::Read) -> Result<Option<Vec<u8>>, ()> {
    let Some(len) = frame_len(line) else {
        return Ok(None);
    };
    if len <= MAX_FRAME_BYTES {
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf).map_err(|_| ())?;
        Ok(Some(buf))
    } else {
        let n = std::io::copy(
            &mut std::io::Read::take(&mut *r, len as u64),
            &mut std::io::sink(),
        )
        .map_err(|_| ())?;
        if n == len as u64 {
            Ok(Some(Vec::new()))
        } else {
            Err(())
        }
    }
}

/// Test encoder (D3b-2 gate; `TALUS_VISION_MOCK=<dir>`): the rows of picture `key` are the SVE1 file `<dir>/<key as 16 hex>.sve`
/// (one record); the pixels are only checked for size. It copies the rows into device memory it owns, on the request's stream,
/// and records `done` there, so the engine path (event wait, device runs, release) is the real one. Not a shipping path.
pub struct MockEncoder {
    dir: std::path::PathBuf,
    n_embd: i64,
    max_rows: i64,
    dev: Vec<*mut c_void>,
    host: Vec<Vec<f32>>,
}

// SAFETY: the device pointers are only used from the serve loop's thread, between encode() and release().
unsafe impl Send for MockEncoder {}

impl MockEncoder {
    pub fn new(dir: impl Into<std::path::PathBuf>, n_embd: i64, max_rows: i64) -> Self {
        Self {
            dir: dir.into(),
            n_embd,
            max_rows,
            dev: Vec::new(),
            host: Vec::new(),
        }
    }

    /// One SVE1 record: (n, nx, ny, rows).
    fn read(&self, key: u64) -> Result<(i64, i64, i64, Vec<f32>), String> {
        let p = self.dir.join(format!("{key:016x}.sve"));
        let b = std::fs::read(&p).map_err(|e| format!("mock encoder: {}: {e}", p.display()))?;
        if b.len() < 20 {
            return Err(format!("mock encoder: {}: short", p.display()));
        }
        let h: Vec<i32> = (0..5)
            .map(|i| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();
        let (n, nx, ny, w) = (h[1] as i64, h[2] as i64, h[3] as i64, h[4] as i64);
        if h[0] != 0x3145_5653
            || w != self.n_embd
            || n != nx * ny
            || b.len() != 20 + (n * w) as usize * 4
        {
            return Err(format!(
                "mock encoder: {}: not one SVE1 record of width {}",
                p.display(),
                self.n_embd
            ));
        }
        Ok((
            n,
            nx,
            ny,
            b[20..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ))
    }
}

impl ImageEncoder for MockEncoder {
    fn n_embd(&self) -> i64 {
        self.n_embd
    }
    fn max_rows(&self) -> i64 {
        self.max_rows
    }
    fn encode(
        &mut self,
        imgs: &[PixelImage],
        stream: RawStream,
        done: RawEvent,
    ) -> Result<Vec<EncodedImage>, String> {
        self.release();
        let mut out = Vec::new();
        for im in imgs {
            if im.rgb.len() != im.w as usize * im.h as usize * 3 {
                return Err("mock encoder: pixel bytes do not match the size".into());
            }
            let (n, nx, ny, rows) = self.read(im.key)?;
            let bytes = rows.len() * 4;
            let mut d: *mut c_void = std::ptr::null_mut();
            // SAFETY: plain HIP runtime calls; `rows` stays alive in self.host until release() (the copy is async).
            unsafe {
                if hip_sys::hipMalloc(&mut d, bytes) != hip_sys::hipSuccess {
                    return Err("mock encoder: device allocation failed".into());
                }
                self.dev.push(d);
                if hip_sys::hipMemcpyAsync(
                    d,
                    rows.as_ptr().cast(),
                    bytes,
                    hip_sys::hipMemcpyHostToDevice,
                    stream as _,
                ) != hip_sys::hipSuccess
                {
                    return Err("mock encoder: upload failed".into());
                }
            }
            self.host.push(rows);
            out.push(EncodedImage {
                rows: d as *const f32,
                n,
                nx,
                ny,
            });
        }
        // SAFETY: as above
        if unsafe { hip_sys::hipEventRecord(done as _, stream as _) } != hip_sys::hipSuccess {
            return Err("mock encoder: event record failed".into());
        }
        Ok(out)
    }
    fn release(&mut self) {
        if self.dev.is_empty() {
            return;
        }
        // SAFETY: the engine calls release() after the prompt pass; free waits for the device to finish with the buffers
        unsafe {
            hip_sys::hipDeviceSynchronize();
            for d in self.dev.drain(..) {
                hip_sys::hipFree(d);
            }
        }
        self.host.clear();
    }
}

/// The `tk_vision_*` group (src/vision/tk_vision.h); aliases so the struct fields stay readable.
type FnVisionLoad = unsafe extern "C" fn(
    *const core::ffi::c_char,
    core::ffi::c_long,
    *mut core::ffi::c_char,
    usize,
) -> *mut c_void;
type FnVisionEncode = unsafe extern "C" fn(
    *mut c_void,
    *const u8,
    core::ffi::c_long,
    core::ffi::c_long,
    *mut f32,
    *mut core::ffi::c_long,
    *mut core::ffi::c_long,
    *mut core::ffi::c_char,
    usize,
) -> core::ffi::c_long;
type FnVisionEncodeRaw = unsafe extern "C" fn(
    *mut c_void,
    *const u8,
    core::ffi::c_long,
    *mut *const f32,
    *mut core::ffi::c_long,
    *mut core::ffi::c_long,
    *mut core::ffi::c_long,
    *mut core::ffi::c_char,
    usize,
) -> core::ffi::c_long;
type FnVisionDestroy = unsafe extern "C" fn(*mut c_void);

/// Native in-engine vision tower: the Strata vision tower compiled into
/// libtalus_kernels.so behind `tk_vision_*` (src/vision/tk_vision.h), resolved with dlsym exactly
/// as the other tk_* groups are.  The tower reads its rows back to the host itself, so this side
/// owns what the trait asks for: the device buffers, the stream ordering, and the `done` event —
/// the same discipline [`MockEncoder`] shows, minus the disk.
pub struct NativeEncoder {
    tower: *mut c_void,
    n_embd: i64,
    max_rows: i64,
    align: i64,
    /// the mmproj's token bounds (tk_vision_info out8[6]/[7]): `VGEOM` answers them so a server's
    /// geometry predictor takes its constants from the engine instead of guessing them
    min_tokens: i64,
    max_tokens: i64,
    device_bytes: u64,
    dev: Vec<*mut c_void>,
    host: Vec<Vec<f32>>,
    /// `TALUS_VISION_DUMP=<dir>`: write each encode's rows as an SVE1 record (`{key:016x}.sve`,
    /// the same file the mock encoder reads) — the C1 parity harness compares those against the
    /// reference tower's records bit for bit, and the dumps run the engine path through the mock.
    dump: Option<std::path::PathBuf>,
    encode_c: FnVisionEncode,
    /// absent on a libtalus_kernels.so built without tk_vision_encode_raw: @raw then refuses
    /// tower does, and @px keeps working
    encode_raw_c: Option<FnVisionEncodeRaw>,
    destroy: FnVisionDestroy,
}

// SAFETY: the device pointers and the tower handle are only used from the serve loop's thread,
// between encode() and release() (as for MockEncoder; the tower itself is touched nowhere else).
unsafe impl Send for NativeEncoder {}

/// The tower's error-buffer size (`TK_VISION_ERRLEN`).
const VISION_ERRLEN: usize = 256;

impl NativeEncoder {
    /// Load the mmproj and warm up the tower (a few seconds; the caller does it once at engine
    /// start).  A libtalus_kernels.so without the tower, or a refused mmproj, is an Err with the
    /// tower's own reason — the serve loop then answers pixel requests honestly rather than
    /// dying.  `max_rows` is the per-request row cap (the engine's ERR line above it).
    pub fn open(mmproj: &std::path::Path, max_rows: i64) -> Result<Self, String> {
        let path = match std::ffi::CString::new(mmproj.to_string_lossy().as_bytes()) {
            Ok(p) => p,
            Err(_) => return Err("native vision: mmproj path contains a NUL".into()),
        };
        let mut err = [0 as core::ffi::c_char; VISION_ERRLEN];
        let mut info = [0 as core::ffi::c_long; 8];
        // SAFETY: dlsym on the process's global symbols (the library was dlopen'ed RTLD_GLOBAL),
        // then the tower's load/info calls; only the encode/destroy pointers outlive this block.
        unsafe {
            use crate::qwen4exp::ffi::load_sym_from as l;
            let n = std::ptr::null_mut();
            let load: FnVisionLoad = l(n, "tk_vision_load")?;
            let info_f: unsafe extern "C" fn(
                *const c_void,
                *mut core::ffi::c_long,
            ) -> core::ffi::c_long = l(n, "tk_vision_info")?;
            let encode_c: FnVisionEncode = l(n, "tk_vision_encode")?;
            // optional by design: an older libtalus_kernels.so has no @raw, and the tower must
            // still load (@px works) rather than die on a missing symbol
            let encode_raw_c: Option<FnVisionEncodeRaw> = l(n, "tk_vision_encode_raw").ok();
            let destroy: FnVisionDestroy = l(n, "tk_vision_destroy")?;
            let tower = load(path.as_ptr(), 0, err.as_mut_ptr(), VISION_ERRLEN);
            if tower.is_null() {
                return Err(format!("native vision: {}", cstr(&err)));
            }
            let device_bytes = info_f(tower, info.as_mut_ptr()).max(0) as u64;
            Ok(Self {
                tower,
                n_embd: info[0],
                max_rows,
                align: info[5],
                min_tokens: info[6],
                max_tokens: info[7],
                device_bytes,
                dev: Vec::new(),
                host: Vec::new(),
                dump: std::env::var_os("TALUS_VISION_DUMP").map(std::path::PathBuf::from),
                encode_c,
                encode_raw_c,
                destroy,
            })
        }
    }

    /// Device bytes held at load (weights + arena).  On Strix Halo's unified memory these are
    /// host pages the pressure guard must count before the first request: the engine raises its
    /// `MemAvailable` floor by [`Self::floor_claim`].
    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// What the tower takes off the host's available memory, plus the tower's host-side working
    /// set during an encode (prepared CHW, the returned rows, and one frame at most): rounded to
    /// MiB to keep a noisy number out of the floor.
    pub fn floor_claim(&self) -> u64 {
        self.device_bytes.div_ceil(256 * 1024 * 1024) * 256 * 1024 * 1024 + 512 * 1024 * 1024
    }
}

impl Drop for NativeEncoder {
    fn drop(&mut self) {
        // SAFETY: the serve loop drops the encoder after its last request completed and released.
        unsafe { (self.destroy)(self.tower) }
    }
}

/// Render the tower's C error buffer (an errlen-sized char array) as a String.
/// # Safety
/// `b` must be a NUL-terminated C string within its array.
fn cstr(b: &[core::ffi::c_char]) -> String {
    // SAFETY: the tower fills it with snprintf, so it is a valid C string.
    unsafe {
        std::ffi::CStr::from_ptr(b.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

/// `TALUS_VISION_DUMP`: one SVE1 record (the mock encoder's file format, named by the key).
/// Best-effort: a failed dump warns and never fails the request.
fn dump_sve(dir: &std::path::Path, key: u64, n: i64, nx: i64, ny: i64, n_embd: i64, rows: &[f32]) {
    let p = dir.join(format!("{key:016x}.sve"));
    let res = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(&p)?;
        let hdr = [
            (0x3145_5653i32, 4),
            (n as i32, 4),
            (nx as i32, 4),
            (ny as i32, 4),
            (n_embd as i32, 4),
        ];
        for (v, _) in hdr {
            f.write_all(&v.to_le_bytes())?;
        }
        for r in rows {
            f.write_all(&r.to_le_bytes())?;
        }
        f.flush()
    })();
    if let Err(e) = res {
        eprintln!("native vision: dump {p:?} failed: {e}");
    }
}

impl ImageEncoder for NativeEncoder {
    fn n_embd(&self) -> i64 {
        self.n_embd
    }
    fn max_rows(&self) -> i64 {
        self.max_rows
    }
    fn encode(
        &mut self,
        imgs: &[PixelImage],
        stream: RawStream,
        done: RawEvent,
    ) -> Result<Vec<EncodedImage>, String> {
        self.release();
        let mut out = Vec::new();
        let mut err = [0 as core::ffi::c_char; VISION_ERRLEN];
        for im in imgs {
            if im.rgb.len() != im.w as usize * im.h as usize * 3 {
                return Err("native vision: pixel bytes do not match the size".into());
            }
            let (w, h) = (im.w as i64, im.h as i64);
            if w % self.align != 0 || h % self.align != 0 {
                return Err(format!(
                    "native vision: picture {w}x{h} is not a multiple of the {} grid (server and \
                     tower must agree on smart_resize)",
                    self.align
                ));
            }
            let want = (w / self.align) * (h / self.align);
            let mut rows = vec![0f32; want as usize * self.n_embd as usize];
            let mut nx = 0;
            let mut ny = 0;
            // SAFETY: plain C-ABI call; `rows` is sized for the grid the tower will report for an
            // aligned picture, and stays owned by `self.host` until release() (the upload is async).
            let n = unsafe {
                (self.encode_c)(
                    self.tower,
                    im.rgb.as_ptr(),
                    w,
                    h,
                    rows.as_mut_ptr(),
                    &mut nx,
                    &mut ny,
                    err.as_mut_ptr(),
                    VISION_ERRLEN,
                )
            };
            if n < 0 {
                return Err(cstr(&err));
            }
            if n != want || nx * ny != n {
                return Err(format!(
                    "native vision: tower gave {n} rows ({nx}x{ny}), the resized grid wanted {want}"
                ));
            }
            if let Some(dir) = &self.dump {
                dump_sve(dir, im.key, n, nx, ny, self.n_embd, &rows);
            }
            let bytes = rows.len() * 4;
            let mut d: *mut c_void = std::ptr::null_mut();
            // SAFETY: plain HIP runtime calls, as in MockEncoder's encode.
            unsafe {
                if hip_sys::hipMalloc(&mut d, bytes) != hip_sys::hipSuccess {
                    return Err("native vision: device allocation failed".into());
                }
                self.dev.push(d);
                if hip_sys::hipMemcpyAsync(
                    d,
                    rows.as_ptr().cast(),
                    bytes,
                    hip_sys::hipMemcpyHostToDevice,
                    stream as _,
                ) != hip_sys::hipSuccess
                {
                    return Err("native vision: upload failed".into());
                }
            }
            self.host.push(rows);
            out.push(EncodedImage {
                rows: d as *const f32,
                n,
                nx,
                ny,
            });
        }
        // SAFETY: as in MockEncoder's encode
        if unsafe { hip_sys::hipEventRecord(done as _, stream as _) } != hip_sys::hipSuccess {
            return Err("native vision: event record failed".into());
        }
        Ok(out)
    }
    fn geom(&self) -> Option<(i64, i64, i64)> {
        Some((self.align, self.min_tokens, self.max_tokens))
    }
    fn encode_raw(
        &mut self,
        raws: &[RawImage],
        stream: RawStream,
        done: RawEvent,
    ) -> Result<Vec<EncodedImage>, String> {
        let encode_raw_c = match self.encode_raw_c {
            Some(f) => f,
            None => {
                return Err(
                    "native vision: libtalus_kernels.so has no tk_vision_encode_raw (@raw \
                     unsupported)"
                        .into(),
                )
            }
        };
        self.release();
        let mut out = Vec::new();
        let mut err = [0 as core::ffi::c_char; VISION_ERRLEN];
        for im in raws {
            let mut rows_out: *const f32 = std::ptr::null();
            let mut nfloats = 0;
            let mut nx = 0;
            let mut ny = 0;
            // SAFETY: plain C-ABI call; *rows_out points into the tower's scratch (see below).
            let n = unsafe {
                encode_raw_c(
                    self.tower,
                    im.data.as_ptr(),
                    im.data.len() as core::ffi::c_long,
                    &mut rows_out,
                    &mut nfloats,
                    &mut nx,
                    &mut ny,
                    err.as_mut_ptr(),
                    VISION_ERRLEN,
                )
            };
            if n < 0 {
                return Err(cstr(&err));
            }
            if nfloats as usize != n as usize * self.n_embd as usize || nx * ny != n {
                return Err(format!(
                    "native vision: @raw gave {n} rows ({nx}x{ny}) but {nfloats} floats"
                ));
            }
            if let Some((pw, ph)) = im.predict {
                if nx * self.align != pw as i64 || ny * self.align != ph as i64 {
                    return Err(format!(
                        "native vision: geometry drift - server predicted a {pw}x{ph} picture \
                         ({} cells), tower prepared {}x{} ({nx}x{ny} cells); server and tower \
                         must agree on smart_resize",
                        pw as i64 / self.align * (ph as i64 / self.align),
                        nx * self.align,
                        ny * self.align,
                    ));
                }
            }
            if let Some(dir) = &self.dump {
                let rows = unsafe { std::slice::from_raw_parts(rows_out, nfloats as usize) };
                dump_sve(dir, im.key, n, nx, ny, self.n_embd, rows);
            }
            let bytes = nfloats as usize * 4;
            let mut d: *mut c_void = std::ptr::null_mut();
            // SAFETY: plain HIP runtime calls, as in MockEncoder's encode.  The copy is
            // SYNCED here, unlike @px's async upload: the tower's scratch is reused by the
            // next encode_raw, so the DMA must be done with these bytes before that.  The
            // prompt path waits on `done` (below) either way.
            unsafe {
                if hip_sys::hipMalloc(&mut d, bytes) != hip_sys::hipSuccess {
                    return Err("native vision: device allocation failed".into());
                }
                self.dev.push(d);
                if hip_sys::hipMemcpy(d, rows_out.cast(), bytes, hip_sys::hipMemcpyHostToDevice)
                    != hip_sys::hipSuccess
                {
                    return Err("native vision: upload failed".into());
                }
            }
            out.push(EncodedImage {
                rows: d as *const f32,
                n,
                nx,
                ny,
            });
        }
        // SAFETY: as in encode()
        if unsafe { hip_sys::hipEventRecord(done as _, stream as _) } != hip_sys::hipSuccess {
            return Err("native vision: event record failed".into());
        }
        Ok(out)
    }
    fn release(&mut self) {
        if self.dev.is_empty() {
            return;
        }
        // SAFETY: the engine calls release() after the prompt pass; free waits for the device to
        // finish with the buffers (and with the host staging behind them)
        unsafe {
            hip_sys::hipDeviceSynchronize();
            for d in self.dev.drain(..) {
                hip_sys::hipFree(d);
            }
        }
        self.host.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_takes_frames_raw_and_stays_in_sync() {
        use std::io::BufRead;
        // a pixel request whose frame bytes look like protocol lines, an oversize one, then a plain line
        let mut input = Vec::new();
        input.extend_from_slice(b"GENI 8 img_keys=1 img_px=2x2 @px 1,2\n");
        input.extend_from_slice(b"STOP\nQUIT\n"); // 10 of the 12 frame bytes
        input.extend_from_slice(b"ab");
        input.extend_from_slice(b"GENI 8 img_keys=1 img_px=2049x2048 @px 1\n");
        input.extend(std::iter::repeat_n(0u8, 2049 * 2048 * 3));
        input.extend_from_slice(b"GEN 4 1,2\n");
        let mut r = std::io::BufReader::new(&input[..]);
        let mut got = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            if r.read_line(&mut line).unwrap() == 0 {
                break;
            }
            let l = line.trim_end_matches('\n').to_string();
            let fr = read_frame(&l, &mut r).unwrap();
            got.push((l, fr.map(|f| f.len())));
        }
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].1, Some(12));
        assert_eq!(got[1].1, Some(0)); // over the cap: read and dropped
        assert_eq!(got[2], ("GEN 4 1,2".to_string(), None));
        // the input ends inside a frame: an error (the reader treats it as the end of input)
        let mut short = std::io::BufReader::new(&b"abc"[..]);
        assert!(read_frame("GENI 8 img_px=2x2 @px 1", &mut short).is_err());
    }

    #[test]
    fn img_px_and_frame_length() {
        assert_eq!(
            parse_img_px("4x2,1x1").unwrap(),
            PixelSpec {
                sizes: vec![(4, 2), (1, 1)],
                bytes: 27
            }
        );
        for bad in [
            "", "4x", "x2", "4x2,", "0x5", "4x2x1", "4X2", "-1x2", "65536x1", "4x2;1x1", "a x2",
        ] {
            assert!(parse_img_px(bad).is_none(), "{bad}");
        }
        assert_eq!(
            frame_len("GENI 8 img_keys=ab img_px=4x2 @px 1,2,3"),
            Some(24)
        );
        assert_eq!(
            frame_len("GENI 8 cache=0 img_keys=ab,cd img_px=4x2,2x2 @px 1,2"),
            Some(36)
        );
        assert_eq!(frame_len("GENI 8 /tmp/a.sve 1,2,3"), None); // SVE form
        assert_eq!(frame_len("GENI 8 img_px=4x2 /tmp/a.sve 1,2"), None); // path, not @px
        assert_eq!(frame_len("GEN 8 img_px=4x2 @px 1,2"), None);
        assert_eq!(frame_len("GENI 8 img_px=bad @px 1,2"), None); // malformed: no frame can be framed
        assert_eq!(frame_len("GENI 8 @px 1,2"), None); // no img_px: ERR, no frame
        assert_eq!(
            frame_len("GENI 8 img_px=2048x2048 @px 1"),
            Some(MAX_FRAME_BYTES)
        );
        // the @raw form frames its list of encoded pictures by img_bytes
        assert_eq!(
            frame_len("GENI 8 img_keys=ab,cd img_bytes=100,200 @raw 1,2"),
            Some(300)
        );
        assert_eq!(
            frame_len("GENI 8 img_keys=ab img_bytes=500 img_px=480x320 @raw 1"),
            Some(500)
        ); // prediction rides along
        assert_eq!(frame_len("GENI 8 img_keys=ab @raw 1"), None); // no img_bytes: ERR, no frame
        assert_eq!(frame_len("GENI 8 img_bytes=ab @raw 1"), None); // malformed
        assert_eq!(parse_img_bytes("10,20"), Some(vec![10, 20]));
        assert_eq!(parse_img_bytes("0"), None);
        assert_eq!(parse_img_bytes(""), None);
    }
}
