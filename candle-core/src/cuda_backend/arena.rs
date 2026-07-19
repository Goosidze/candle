//! Flat Arena — статический bump аллокатор для CUDA Graphs
//! Минимальная версия для компиляции без tracing

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, RwLock,
};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use crate::{Result, Error};

const ALIGNMENT: usize = 256;

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

struct CachedEntry {
    ptr: u64,
    bytes: usize,
    _keep_alive: Arc<dyn Send + Sync>,
}

pub struct FlatArena {
    stream: Arc<CudaStream>,
    giant: Option<GiantBlock>,
    offset: AtomicUsize,
    is_frozen: AtomicBool,
    alloc_seq: AtomicUsize,
    cache: RwLock<HashMap<usize, CachedEntry>>,
    enabled: AtomicBool,
}

struct GiantBlock {
    base_ptr: u64,
    size: usize,
    // Keep alive via ManuallyDrop slice that we will free manually on Drop
    _holder: Arc<std::mem::ManuallyDrop<CudaSlice<u8>>>,
}

impl FlatArena {
    pub fn new(stream: Arc<CudaStream>, giant_bytes: usize) -> Result<Self> {
        let enabled = std::env::var("SENIOR_AGENT_STATIC_ARENA")
            .map(|v| !v.is_empty() && v != "0" && !v.to_lowercase().contains("false"))
            .unwrap_or(false);

        if !enabled || giant_bytes == 0 {
            return Ok(Self {
                stream,
                giant: None,
                offset: AtomicUsize::new(0),
                is_frozen: AtomicBool::new(false),
                alloc_seq: AtomicUsize::new(0),
                cache: RwLock::new(HashMap::new()),
                enabled: AtomicBool::new(false),
            });
        }

        // Allocate giant block
        let giant_slice: CudaSlice<u8> = unsafe { stream.alloc::<u8>(giant_bytes) }
            .map_err(|e| Error::Msg(format!("arena giant alloc {giant_bytes} failed: {e}")))?;

        // Leak to get raw ptr, then upgrade back to holder that won't free twice
        // SAFETY: CudaSlice first field is CUdeviceptr (u64)
        let base_ptr = unsafe {
            let ptr_u64 = std::ptr::read(&giant_slice as *const CudaSlice<u8> as *const u64);
            // Prevent original slice from freeing
            std::mem::forget(giant_slice);
            ptr_u64
        };

        // Create holder that will free on Drop of arena (via cuMemFree manually)
        let holder_slice: CudaSlice<u8> = unsafe { stream.upgrade_device_ptr(base_ptr, giant_bytes) };
        let holder = Arc::new(std::mem::ManuallyDrop::new(holder_slice));

        Ok(Self {
            stream,
            giant: Some(GiantBlock {
                base_ptr,
                size: giant_bytes,
                _holder: holder,
            }),
            offset: AtomicUsize::new(0),
            is_frozen: AtomicBool::new(false),
            alloc_seq: AtomicUsize::new(0),
            cache: RwLock::new(HashMap::new()),
            enabled: AtomicBool::new(true),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
    pub fn is_frozen(&self) -> bool {
        self.is_frozen.load(Ordering::Acquire)
    }
    pub fn seq(&self) -> usize {
        self.alloc_seq.load(Ordering::Relaxed)
    }
    pub fn offset(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }
    pub fn cache_len(&self) -> usize {
        self.cache.read().unwrap().len()
    }
    pub fn freeze(&self) {
        self.is_frozen.store(true, Ordering::Release);
    }
    pub fn unfreeze(&self) {
        self.is_frozen.store(false, Ordering::Release);
        self.alloc_seq.store(0, Ordering::Release);
    }
    pub fn owns(&self, ptr_u64: u64) -> bool {
        if let Some(giant) = &self.giant {
            ptr_u64 >= giant.base_ptr && ptr_u64 < giant.base_ptr + giant.size as u64
        } else {
            let cache = self.cache.read().unwrap();
            cache
                .values()
                .any(|e| ptr_u64 >= e.ptr && ptr_u64 < e.ptr + e.bytes as u64)
        }
    }

    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let aligned_bytes = align_up(bytes, ALIGNMENT);
        let seq = self.alloc_seq.fetch_add(1, Ordering::Relaxed);

        if self.is_frozen() {
            let cache = self.cache.read().unwrap();
            if let Some(entry) = cache.get(&seq) {
                if entry.bytes < aligned_bytes {
                    crate::bail!(
                        "FlatArena frozen miss: seq={} expected bytes={} got cached bytes={} — forward not deterministic",
                        seq,
                        aligned_bytes,
                        entry.bytes
                    );
                }
                let slice: CudaSlice<T> = self.stream.upgrade_device_ptr(entry.ptr, len);
                return Ok(slice);
            } else {
                crate::bail!(
                    "FlatArena frozen cache miss: seq={} bytes={} — forward not deterministic",
                    seq,
                    aligned_bytes
                );
            }
        }

        // Not frozen — normal alloc (NOT from giant, giant is only for explicit KV pool allocs)
        // This fixes model load OOM / invalid argument because model weights (9GB) don't fit into 2GB giant
        let slice: CudaSlice<T> = self
            .stream
            .alloc::<T>(len)
            .map_err(|e| Error::Msg(format!("arena cache alloc failed: {e}")))?;
        let ptr = unsafe { std::ptr::read(&slice as *const CudaSlice<T> as *const u64) };
        // Keep alive via upgraded u8 slice in Arc
        let slice_u8: CudaSlice<u8> = unsafe { self.stream.upgrade_device_ptr(ptr, aligned_bytes) };
        let keep_alive: Arc<dyn Send + Sync> =
            Arc::new(std::mem::ManuallyDrop::new(slice_u8)) as Arc<dyn Send + Sync>;
        self.cache.write().unwrap().insert(
            seq,
            CachedEntry {
                ptr,
                bytes: aligned_bytes,
                _keep_alive: keep_alive,
            },
        );
        unsafe {
            let leaked_ptr = std::ptr::read(&slice as *const CudaSlice<T> as *const u64);
            std::mem::forget(slice);
            Ok(self.stream.upgrade_device_ptr(leaked_ptr, len))
        }
    }

    /// Explicit giant bump alloc — only for KV pools (k_pages/v_pages/static_out_buf)
    /// These never get freed individually, only when giant block freed
    pub unsafe fn alloc_from_giant<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let aligned_bytes = align_up(bytes, ALIGNMENT);
        if let Some(giant) = &self.giant {
            let current_offset = self.offset.fetch_add(aligned_bytes, Ordering::Relaxed);
            let aligned_offset = align_up(current_offset, ALIGNMENT);
            if aligned_offset + aligned_bytes > giant.size {
                crate::bail!(
                    "FlatArena giant OOM: offset {} + {} > size {}",
                    aligned_offset,
                    aligned_bytes,
                    giant.size
                );
            }
            let ptr = giant.base_ptr + aligned_offset as u64;
            let slice: CudaSlice<T> = self.stream.upgrade_device_ptr(ptr, len);
            // For giant allocs we don't need to cache for frozen (they are fixed anyway)
            // But we also don't want them to free individually, so we keep giant holder alive
            Ok(slice)
        } else {
            // No giant — fallback to normal alloc
            self.alloc::<T>(len)
        }
    }

    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
        self.alloc_seq.store(0, Ordering::Relaxed);
        self.is_frozen.store(false, Ordering::Release);
        self.cache.write().unwrap().clear();
    }
}

impl Drop for FlatArena {
    fn drop(&mut self) {
        if let Some(giant) = &self.giant {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFree_v2(giant.base_ptr);
            }
        }
    }
}
