//! Flat Arena — статический bump аллокатор для CUDA Graphs
//! Фикс: используем DevicePtr::device_ptr() вместо ptr::read (repr(Rust) ловушка)

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, RwLock,
};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};
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
    cache_enabled: AtomicBool,
}

struct GiantBlock {
    base_ptr: u64,
    size: usize,
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
                cache_enabled: AtomicBool::new(true),
            });
        }

        // Allocate giant block once via normal alloc
        let giant_slice: CudaSlice<u8> = unsafe { stream.alloc::<u8>(giant_bytes) }
            .map_err(|e| Error::Msg(format!("arena giant alloc {giant_bytes} failed: {e}")))?;

        // FIX: используем легальный device_ptr() вместо ptr::read (repr(Rust) ловушка!)
        // CudaSlice layout не гарантирован, нельзя читать первое поле как u64
        // Guard должен дропнуться до forget и до move stream
        let base_ptr = {
            let (cu_ptr, guard) = giant_slice.device_ptr(&stream);
            let ptr = cu_ptr as u64;
            drop(guard);
            ptr
        };

        // Держатель который не будет фришить дважды — ManuallyDrop
        // Мы уже получили ptr через device_ptr, теперь забываем оригинальный slice чтобы он не фришил
        // А holder будет фришить через cuMemFree в Drop FlatArena
        unsafe {
            std::mem::forget(giant_slice);
        }
        let holder_slice: CudaSlice<u8> = unsafe { stream.upgrade_device_ptr(base_ptr, giant_bytes) };
        let holder = Arc::new(std::mem::ManuallyDrop::new(holder_slice));

        Ok(Self {
            stream: stream.clone(),
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
            cache_enabled: AtomicBool::new(true),
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
        self.alloc_seq.store(0, Ordering::Release);
        self.is_frozen.store(true, Ordering::Release);
    }

    pub fn unfreeze(&self) {
        self.is_frozen.store(false, Ordering::Release);
        self.alloc_seq.store(0, Ordering::Release);
    }

    pub fn set_cache_enabled(&self, enabled: bool) {
        self.cache_enabled.store(enabled, Ordering::Release);
    }

    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled.load(Ordering::Relaxed)
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

    // General alloc — НЕ из giant, обычный alloc + кеш для детерминизма
    // Это фиксит OOM: веса модели 7.1GB не должны идти в 2GB giant
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

        // Not frozen — check if caching is enabled (disabled during prefill for speed)
        if !self.is_cache_enabled() {
            // Prefill: no caching, direct alloc (8x faster, no HashMap overhead)
            return self
                .stream
                .alloc::<T>(len)
                .map_err(|e| Error::Msg(format!("arena direct alloc failed: {e}")));
        }

        // First decode token: normal alloc + cache ptr via legal device_ptr()
        let slice: CudaSlice<T> = self
            .stream
            .alloc::<T>(len)
            .map_err(|e| Error::Msg(format!("arena cache alloc failed: {e}")))?;
        let ptr = {
            let (cu_ptr, guard) = slice.device_ptr(&self.stream);
            let p = cu_ptr as u64;
            drop(guard);
            p
        };

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
            let ptr_u64 = {
                let (cu_ptr, guard) = slice.device_ptr(&self.stream);
                let p = cu_ptr as u64;
                drop(guard);
                p
            };
            std::mem::forget(slice);
            Ok(self.stream.upgrade_device_ptr(ptr_u64, len))
        }
    }

    // Explicit giant bump — only for KV pools (fixed address for CUDA Graphs)
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
            Ok(slice)
        } else {
            // No giant — fallback to normal cache alloc
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
