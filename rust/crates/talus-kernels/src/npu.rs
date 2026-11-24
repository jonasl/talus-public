//! Safe Rust wrapper for the Talus XDNA2 NPU C ABI (`include/talus_npu.h`).

use core::ffi::{c_char, c_void};
use std::ffi::{CStr, CString};
use std::path::Path;
use talus_core_api::{CoreError, Result};
use talus_kernels_sys::LibTalusKernels;

#[repr(C)]
pub struct talus_npu_ctx {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TalusNpuConfigRaw {
    pub xclbin_path: *const c_char,
    pub insts_path: *const c_char,
    pub tokens: u32,
    pub k: u32,
    pub n: u32,
    pub timeout_ms: u32,
    pub isolate: i32,
    pub keepalive: i32,
    pub allow_unheld_fclk: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TalusNpuStats {
    pub submits: u64,
    pub waits: u64,
    pub slices_run: u64,
    pub encode_us_total: f64,
    pub npu_wait_us_total: f64,
    pub host_wait_us_total: f64,
    pub copy_us_total: f64,
}

/// Configuration for creating a [`NpuContext`].
#[derive(Debug, Clone)]
pub struct NpuConfig {
    pub xclbin_path: String,
    pub insts_path: String,
    pub tokens: u32,
    pub k: u32,
    pub n: u32,
    pub timeout_ms: u32,
    pub isolate: bool,
    pub keepalive: bool,
    pub allow_unheld_fclk: bool,
}

impl NpuConfig {
    pub fn new(xclbin_path: impl Into<String>, insts_path: impl Into<String>) -> Self {
        Self {
            xclbin_path: xclbin_path.into(),
            insts_path: insts_path.into(),
            tokens: 4096,
            k: 2560,
            n: 1024,
            timeout_ms: 8000,
            isolate: false,
            keepalive: true,
            allow_unheld_fclk: true,
        }
    }
}

#[derive(Clone, Copy)]
pub struct NpuFns {
    pub config_default: unsafe extern "C" fn(*mut TalusNpuConfigRaw),
    pub create: unsafe extern "C" fn(
        *const TalusNpuConfigRaw,
        *mut *mut talus_npu_ctx,
        *mut c_char,
        usize,
    ) -> i32,
    pub destroy: unsafe extern "C" fn(*mut talus_npu_ctx),
    pub healthy: unsafe extern "C" fn(*const talus_npu_ctx) -> i32,
    pub tokens: unsafe extern "C" fn(*const talus_npu_ctx) -> u32,
    pub k: unsafe extern "C" fn(*const talus_npu_ctx) -> u32,
    pub n: unsafe extern "C" fn(*const talus_npu_ctx) -> u32,
    pub a_bytes: unsafe extern "C" fn(*const talus_npu_ctx) -> usize,
    pub b_bytes: unsafe extern "C" fn(*const talus_npu_ctx) -> usize,
    pub c_bytes: unsafe extern "C" fn(*const talus_npu_ctx) -> usize,
    pub alloc_buffer: unsafe extern "C" fn(*mut talus_npu_ctx, usize, *mut c_char, usize) -> i32,
    pub host_ptr: unsafe extern "C" fn(*const talus_npu_ctx, i32) -> *mut c_void,
    pub device_ptr: unsafe extern "C" fn(*const talus_npu_ctx, i32) -> *mut c_void,
    pub staged_ptr: unsafe extern "C" fn(*const talus_npu_ctx, i32) -> *mut c_void,
    pub finalize_weights: unsafe extern "C" fn(*mut talus_npu_ctx, i32),
    pub stage_stream: unsafe extern "C" fn(*const talus_npu_ctx) -> *mut c_void,
    pub submit: unsafe extern "C" fn(
        *mut talus_npu_ctx,
        *mut c_void,
        i32,
        *const i32,
        *const i32,
        i32,
    ) -> i32,
    pub wait: unsafe extern "C" fn(*mut talus_npu_ctx, *mut c_void) -> i32,
    pub get_stats: unsafe extern "C" fn(*const talus_npu_ctx, *mut TalusNpuStats),
    pub check_env_safety: unsafe extern "C" fn(*mut c_char, usize) -> i32,
    pub encode_activations_f16:
        unsafe extern "C" fn(*const u16, usize, u32, u32, u32, u32, u32, *mut c_void, *mut c_void),
    pub encode_weights_q8: unsafe extern "C" fn(
        *const c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut c_void,
    ),
    pub encode_weights_q8_pair: unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut c_void,
    ),
    pub encode_weights_f16: unsafe extern "C" fn(
        *const u16,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut c_void,
    ),
    pub encode_weights_bf16: unsafe extern "C" fn(
        *const u16,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut c_void,
    ),
    pub decode_output: unsafe extern "C" fn(
        *const c_void,
        *mut f32,
        u32,
        u32,
        u32,
        usize,
        u32,
        i32,
        u32,
        *mut c_void,
    ),
    pub swiglu: unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *mut u16,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
    ),
    pub swiglu_packed:
        unsafe extern "C" fn(*const c_void, *mut u16, u32, u32, u32, u32, *mut c_void),
}

impl NpuFns {
    #[cfg(unix)]
    pub fn resolve(lib: &LibTalusKernels) -> Result<Self> {
        let map_err = |e: String| CoreError::CapabilityRefused {
            arch: "npu".to_string(),
            reason: e,
        };
        unsafe {
            Ok(Self {
                config_default: lib
                    .resolve_symbol("talus_npu_config_default")
                    .map_err(map_err)?,
                create: lib.resolve_symbol("talus_npu_create").map_err(map_err)?,
                destroy: lib.resolve_symbol("talus_npu_destroy").map_err(map_err)?,
                healthy: lib.resolve_symbol("talus_npu_healthy").map_err(map_err)?,
                tokens: lib.resolve_symbol("talus_npu_tokens").map_err(map_err)?,
                k: lib.resolve_symbol("talus_npu_k").map_err(map_err)?,
                n: lib.resolve_symbol("talus_npu_n").map_err(map_err)?,
                a_bytes: lib.resolve_symbol("talus_npu_a_bytes").map_err(map_err)?,
                b_bytes: lib.resolve_symbol("talus_npu_b_bytes").map_err(map_err)?,
                c_bytes: lib.resolve_symbol("talus_npu_c_bytes").map_err(map_err)?,
                alloc_buffer: lib
                    .resolve_symbol("talus_npu_alloc_buffer")
                    .map_err(map_err)?,
                host_ptr: lib.resolve_symbol("talus_npu_host_ptr").map_err(map_err)?,
                device_ptr: lib
                    .resolve_symbol("talus_npu_device_ptr")
                    .map_err(map_err)?,
                staged_ptr: lib
                    .resolve_symbol("talus_npu_staged_ptr")
                    .map_err(map_err)?,
                finalize_weights: lib
                    .resolve_symbol("talus_npu_finalize_weights")
                    .map_err(map_err)?,
                stage_stream: lib
                    .resolve_symbol("talus_npu_stage_stream")
                    .map_err(map_err)?,
                submit: lib.resolve_symbol("talus_npu_submit").map_err(map_err)?,
                wait: lib.resolve_symbol("talus_npu_wait").map_err(map_err)?,
                get_stats: lib.resolve_symbol("talus_npu_get_stats").map_err(map_err)?,
                check_env_safety: lib
                    .resolve_symbol("talus_npu_check_env_safety")
                    .map_err(map_err)?,
                encode_activations_f16: lib
                    .resolve_symbol("talus_npu_encode_activations_f16")
                    .map_err(map_err)?,
                encode_weights_q8: lib
                    .resolve_symbol("talus_npu_encode_weights_q8")
                    .map_err(map_err)?,
                encode_weights_q8_pair: lib
                    .resolve_symbol("talus_npu_encode_weights_q8_pair")
                    .map_err(map_err)?,
                encode_weights_f16: lib
                    .resolve_symbol("talus_npu_encode_weights_f16")
                    .map_err(map_err)?,
                encode_weights_bf16: lib
                    .resolve_symbol("talus_npu_encode_weights_bf16")
                    .map_err(map_err)?,
                decode_output: lib
                    .resolve_symbol("talus_npu_decode_output")
                    .map_err(map_err)?,
                swiglu: lib.resolve_symbol("talus_npu_swiglu").map_err(map_err)?,
                swiglu_packed: lib
                    .resolve_symbol("talus_npu_swiglu_packed")
                    .map_err(map_err)?,
            })
        }
    }
}

/// Owned handle to a Talus XDNA2 NPU context.
pub struct NpuContext {
    fns: NpuFns,
    ctx: *mut talus_npu_ctx,
}

unsafe impl Send for NpuContext {}
unsafe impl Sync for NpuContext {}

impl NpuContext {
    pub fn check_env_safety(fns: &NpuFns) -> (bool, String) {
        let mut buf = [0 as c_char; 256];
        let safe = unsafe { (fns.check_env_safety)(buf.as_mut_ptr(), buf.len()) } != 0;
        let msg = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        (safe, msg)
    }

    pub fn create(fns: NpuFns, cfg: &NpuConfig) -> Result<Self> {
        let cx =
            CString::new(cfg.xclbin_path.as_str()).map_err(|e| CoreError::InvalidArgument {
                reason: e.to_string(),
            })?;
        let ci = CString::new(cfg.insts_path.as_str()).map_err(|e| CoreError::InvalidArgument {
            reason: e.to_string(),
        })?;
        let mut raw = TalusNpuConfigRaw {
            xclbin_path: cx.as_ptr(),
            insts_path: ci.as_ptr(),
            tokens: cfg.tokens,
            k: cfg.k,
            n: cfg.n,
            timeout_ms: cfg.timeout_ms,
            isolate: i32::from(cfg.isolate),
            keepalive: i32::from(cfg.keepalive),
            allow_unheld_fclk: i32::from(cfg.allow_unheld_fclk),
        };
        let mut ctx: *mut talus_npu_ctx = std::ptr::null_mut();
        let mut err = [0 as c_char; 512];
        let rc = unsafe { (fns.create)(&mut raw, &mut ctx, err.as_mut_ptr(), err.len()) };
        if rc != 0 || ctx.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(CoreError::CapabilityRefused {
                arch: "xdna2".to_string(),
                reason: if msg.is_empty() {
                    format!("talus_npu_create failed with rc={rc}")
                } else {
                    msg
                },
            });
        }
        Ok(Self { fns, ctx })
    }

    pub fn healthy(&self) -> bool {
        unsafe { (self.fns.healthy)(self.ctx) != 0 }
    }

    pub fn shape(&self) -> (u32, u32, u32) {
        unsafe {
            (
                (self.fns.tokens)(self.ctx),
                (self.fns.k)(self.ctx),
                (self.fns.n)(self.ctx),
            )
        }
    }

    pub fn buffer_sizes(&self) -> (usize, usize, usize) {
        unsafe {
            (
                (self.fns.a_bytes)(self.ctx),
                (self.fns.b_bytes)(self.ctx),
                (self.fns.c_bytes)(self.ctx),
            )
        }
    }

    pub fn alloc_buffer(&self, bytes: usize) -> Result<i32> {
        let mut err = [0 as c_char; 256];
        let id = unsafe { (self.fns.alloc_buffer)(self.ctx, bytes, err.as_mut_ptr(), err.len()) };
        if id < 0 {
            let _msg = unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(CoreError::OutOfMemory {
                requested_bytes: bytes as u64,
                available_bytes: 0,
            });
        }
        Ok(id)
    }

    pub fn stats(&self) -> TalusNpuStats {
        let mut s = TalusNpuStats::default();
        unsafe { (self.fns.get_stats)(self.ctx, &mut s) };
        s
    }

    pub fn fns(&self) -> &NpuFns {
        &self.fns
    }

    pub fn raw(&self) -> *mut talus_npu_ctx {
        self.ctx
    }
}

impl Drop for NpuContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { (self.fns.destroy)(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// Compute `v8bfp16ebs8` byte size for `rows * cols` elements (`cols % 8 == 0`).
pub const fn v8bfp16ebs8_bytes(rows: usize, cols: usize) -> usize {
    (rows * cols / 8) * 9
}

/// Resolve default `final_4096x2560x1024_128x64x128.xclbin` and instruction paths from `dir`.
pub fn default_kernel_paths(dir: impl AsRef<Path>) -> (String, String) {
    let d = dir.as_ref();
    (
        d.join("final_4096x2560x1024_128x64x128.xclbin")
            .to_string_lossy()
            .into_owned(),
        d.join("insts_4096x2560x1024_128x64x128.txt")
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v8bfp16ebs8_byte_counts_match_4096x2560x1024() {
        assert_eq!(v8bfp16ebs8_bytes(4096, 2560), 11_796_480);
        assert_eq!(v8bfp16ebs8_bytes(2560, 1024), 2_949_120);
        assert_eq!(v8bfp16ebs8_bytes(4096, 1024), 4_718_592);
    }

    #[test]
    fn npu_config_defaults() {
        let cfg = NpuConfig::new("/tmp/a.xclbin", "/tmp/i.txt");
        assert_eq!(cfg.tokens, 4096);
        assert_eq!(cfg.k, 2560);
        assert_eq!(cfg.n, 1024);
        assert_eq!(cfg.timeout_ms, 8000);
        assert!(!cfg.isolate);
        assert!(cfg.keepalive);
        assert!(cfg.allow_unheld_fclk);
    }

    #[test]
    fn c_abi_struct_sizes_match_c_header() {
        assert_eq!(
            std::mem::size_of::<TalusNpuStats>(),
            3 * std::mem::size_of::<u64>() + 4 * std::mem::size_of::<f64>()
        );
    }

    #[test]
    fn resolve_npu_symbols_if_lib_present() {
        let candidates = [
            "build/libtalus_kernels.so",
            "../../../build/libtalus_kernels.so",
        ];
        let Some(so) = candidates.iter().find(|p| Path::new(p).exists()) else {
            return;
        };
        let lib = LibTalusKernels::load(so).expect("load libtalus_kernels.so");
        let fns = NpuFns::resolve(&lib).expect("resolve talus_npu_* symbols");
        let (_safe, diag) = NpuContext::check_env_safety(&fns);
        assert!(!diag.is_empty());
    }
}
