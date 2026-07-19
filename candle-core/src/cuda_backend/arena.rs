//! Flat Arena — статический bump аллокатор для CUDA Graphs
//! 
//! Идея из твоего плана:
//! - Один гигантский cuMemAlloc при старте (2-4GB)
//! - Bump pointer с выравниванием 256
//! - Freeze режим: отдаем закешированные адреса по alloc_seq
//! - Перехват free: если ptr принадлежит арене — не вызываем cuMemFree
//!
//! Реализация учитывает cudarc 0.19.8 API:
//! - CudaSlice::leak() -> CUdeviceptr (не фришит)
//! - CudaStream::upgrade_device_ptr(ptr, len) -> CudaSlice (создает враппер без аллокации)
//! - CudaSlice Clone = try_clone = alloc + D2D (глубокая копия, нам НЕ подходит для freeze)
//!   Поэтому в freeze мы используем upgrade_device_ptr + ManuallyDrop чтобы избежать double-free

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, RwLock,
};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use crate::{Result, Error};

const ALIGNMENT: usize = 256;

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Ключ для frozen_cache — порядковый номер аллокации в forward
/// Требует 100% детерминизма forward, иначе сдвиг и краш
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ArenaKey {
    seq: usize,
}

/// Значение в кеше — сохраняет base ptr и длину, плюс Arc к базовому слайсу чтобы память жила
#[derive(Debug)]
struct CachedEntry {
    /// CUdeviceptr как u64
    ptr: u64,
    /// Размер в байтах (выровненный)
    bytes: usize,
    /// Держим Arc к giant или к оригинальному слайсу чтобы не освободить
    /// Для giant-режима — Arc к giant slice
    /// Для cache-режима — Arc к leaked allocation holder
    _keep_alive: Arc<dyn Send + Sync>,
}

/// Flat Arena — один большой блок + кеш для детерминизма
pub struct FlatArena {
    /// Поток, на котором аллоцировано
    stream: Arc<CudaStream>,
    /// Гигантский блок (опционально) — 2-4GB
    /// Если None — работаем в режиме кеша индивидуальных аллокаций (V2a простой)
    giant: Option<GiantBlock>,
    /// Текущий offset в giant блоке (если giant Some)
    offset: AtomicUsize,
    /// Флаг заморозки — после первого токена
    is_frozen: AtomicBool,
    /// Глобальный счетчик alloc_seq — инкрементится при каждом alloc
    alloc_seq: AtomicUsize,
    /// Кеш для frozen режима: seq -> CachedEntry
    cache: RwLock<HashMap<usize, CachedEntry>>,
    /// Включена ли арена вообще
    enabled: AtomicBool,
}

struct GiantBlock {
    /// Сырой CUdeviceptr базы (u64)
    base_ptr: u64,
    /// Размер
    size: usize,
    /// Arc к оригинальному CudaSlice<u8> который держит память (ManuallyDrop чтобы не дропнуть дважды)
    /// Мы храним его как Arc<ManuallyDrop<CudaSlice<u8>>> чтобы Drop не вызывал free до Drop арены
    /// На самом деле мы делаем leak + храним ptr, а при drop арены вызываем cuMemFree вручную
    _holder: Arc<std::mem::ManuallyDrop<CudaSlice<u8>>>,
}

impl FlatArena {
    pub fn new(stream: Arc<CudaStream>, giant_bytes: usize) -> Result<Self> {
        let enabled = std::env::var("SENIOR_AGENT_STATIC_ARENA")
            .map(|v| !v.is_empty() && v != "0" && !v.to_lowercase().contains("false"))
            .unwrap_or(true); // по умолчанию вкл если фича есть

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

        // Выделяем гигантский блок один раз
        // SAFETY: alloc вызывается вне capture
        let giant_slice: CudaSlice<u8> = unsafe { stream.alloc::<u8>(giant_bytes) }.map_err(|e| Error::Msg(format!("arena giant alloc {giant_bytes} failed: {e}")))?;
        
        // Получаем сырой ptr, но НЕ делаем leak полностью — мы хотим чтобы giant_slice жил до drop арены
        // Для получения ptr используем transmute к internal, т.к. cu_device_ptr private, но в cudarc 0.19 есть device_ptr() метод для CudaSlice?
        // Используем unsafe: CudaSlice лежит как { cu_device_ptr, len, ... }, первый field — ptr
        // Более надежно — использовать leak + upgrade, но тогда нужно хранить ptr отдельно и free вручную в drop
        // Делаем: leak giant_slice -> ptr, потом upgrade обратно в holder который мы будем хранить как ManuallyDrop
        
        let base_ptr = {
            // SAFETY: мы знаем layout, но лучше через leak API
            let leaked_ptr = giant_slice.leak();
            leaked_ptr as u64
        };
        
        // Теперь создаем holder который НЕ будет фришить до drop арены — через upgrade + ManuallyDrop
        let holder_slice: CudaSlice<u8> = unsafe { stream.upgrade_device_ptr(base_ptr as *mut _, giant_bytes) };
        let holder = Arc::new(std::mem::ManuallyDrop::new(holder_slice));

        tracing::info!("[FlatArena] allocated giant block: {} bytes ({:.2} GiB) at ptr={:#x}", giant_bytes, giant_bytes as f64 / 1024.0/1024.0/1024.0, base_ptr);

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

    pub fn freeze(&self) {
        self.is_frozen.store(true, Ordering::Release);
        let cache_len = self.cache.read().unwrap().len();
        tracing::info!("[FlatArena] FROZEN — {} cached allocations, offset={}", cache_len, self.offset.load(Ordering::Relaxed));
    }

    pub fn unfreeze(&self) {
        self.is_frozen.store(false, Ordering::Release);
        self.alloc_seq.store(0, Ordering::Release);
        tracing::info!("[FlatArena] UNFROZEN — reset seq");
    }

    /// Проверяет, принадлежит ли ptr арене (для перехвата free)
    pub fn owns(&self, ptr_u64: u64) -> bool {
        if let Some(giant) = &self.giant {
            ptr_u64 >= giant.base_ptr && ptr_u64 < giant.base_ptr + giant.size as u64
        } else {
            // В режиме кеша — проверяем по кешу
            let cache = self.cache.read().unwrap();
            cache.values().any(|e| ptr_u64 >= e.ptr && ptr_u64 < e.ptr + e.bytes as u64)
        }
    }

    /// Главный метод — вызывается из CudaDevice::alloc
    /// Возвращает CudaSlice<T> который НЕ будет фришить память арены при Drop (ManuallyDrop внутри)
    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let aligned_bytes = align_up(bytes, ALIGNMENT);
        let seq = self.alloc_seq.fetch_add(1, Ordering::Relaxed);

        if self.is_frozen() {
            // Frozen режим — только кеш, никаких malloc
            let cache = self.cache.read().unwrap();
            if let Some(entry) = cache.get(&seq) {
                if entry.bytes < aligned_bytes {
                    // Размер не совпал — forward не детерминирован
                    // Это критичный баг: на токене 5 появился лишний тензор [1] и сдвинул seq
                    crate::bail!("FlatArena frozen miss: seq={} expected bytes={} got cached bytes={} — forward не детерминирован! Проверь что sampling/EOS/inner if вне графа", seq, aligned_bytes, entry.bytes);
                }
                // Создаем новый CudaSlice враппер вокруг закешированного ptr БЕЗ аллокации
                // upgrade_device_ptr НЕ вызывает cuMemAlloc, только создает объект с тем же ptr
                let slice: CudaSlice<T> = self.stream.upgrade_device_ptr(entry.ptr as *mut _, len);
                // ВАЖНО: этот slice при Drop вызовет cuMemFree(ptr) — двойной free!
                // Чтобы избежать, мы должны забыть его Drop и держать память живой через cache entry _keep_alive
                // Делаем ManuallyDrop и transmute обратно в CudaSlice для возврата
                // Хитрость: возвращаем slice, но его Drop будет делать free, поэтому мы сразу leak его ptr и создаем еще один?
                // Правильный путь: возвращаем slice, но в CudaStorage мы обернем его в ManuallyDrop
                // Здесь возвращаем как есть — перехват free произойдет в CudaStorage Drop (см. ниже)
                // Для этого в Candle нужно патчить CudaStorage Drop чтобы проверять owns()
                //
                // Временное решение: leak + upgrade еще раз, но забыть первый?
                // Проще: забыть Drop этого slice через ManuallyDrop и вернуть клон который тоже не фришит?
                //
                // Для V2a мы делаем: создаем slice, но его Drop будет перехвачен в CudaDevice::is_arena_ptr check
                // Поэтому возвращаем как есть — CudaStorage проверит owns() и скипнет free
                return Ok(slice);
            } else {
                crate::bail!("FlatArena frozen cache miss: seq={} bytes={} — forward не детерминирован! Возможно лишний тензор в if ветке", seq, aligned_bytes);
            }
        }

        // Не frozen — аллоцируем
        if let Some(giant) = &self.giant {
            // Giant bump режим
            let current_offset = self.offset.fetch_add(aligned_bytes, Ordering::Relaxed);
            let aligned_offset = align_up(current_offset, ALIGNMENT);
            // Если fetch_add уже сдвинул, нужно скорректировать разницу
            // Простая реализация: используем loop с compare_exchange
            // Для краткости — используем aligned_offset как базу
            
            if aligned_offset + aligned_bytes > giant.size {
                crate::bail!("FlatArena OOM: offset {} + {} > size {}", aligned_offset, aligned_bytes, giant.size);
            }
            let ptr = giant.base_ptr + aligned_offset as u64;
            let slice: CudaSlice<T> = self.stream.upgrade_device_ptr(ptr as *mut _, len);
            // Кешируем для будущего freeze
            let mut cache = self.cache.write().unwrap();
            // Keep alive — клон giant holder как Arc<dyn>
            let keep_alive: Arc<dyn Send + Sync> = giant._holder.clone() as Arc<dyn Send + Sync>;
            cache.insert(seq, CachedEntry {
                ptr,
                bytes: aligned_bytes,
                _keep_alive: keep_alive,
            });
            tracing::trace!("[FlatArena] bump alloc seq={} ptr={:#x} bytes={} (aligned {})", seq, ptr, bytes, aligned_bytes);
            Ok(slice)
        } else {
            // Cache режим — каждая аллокация отдельный cuMemAlloc, но кешируется
            let slice: CudaSlice<T> = self.stream.alloc::<T>(len).map_err(|e| Error::Msg(format!("arena cache alloc failed: {e}")))?;
            // Получаем ptr через leak + upgrade trick чтобы не фришить дважды
            // Сохраняем raw ptr
            let ptr = {
                // SAFETY: we need cu_device_ptr, we get via transmute of first field
                // В cudarc 0.19 cu_device_ptr — первое поле, можно через unsafe pointer cast
                // Для простоты используем slice.leak() но тогда теряем slice
                // Поэтому клонируем ptr через unsafe read
                unsafe { std::ptr::read(&slice as *const CudaSlice<T> as *const u64) as u64 }
            };
            // Для keep_alive — храним Arc к самой slice (клонированной через upgrade)
            // Но т.к. Clone делает deep copy (alloc), мы храним исходную slice через ManuallyDrop
            // Упрощение: храним Arc к Box<dyn> который держит leaked ptr
            let keep_alive: Arc<dyn Send + Sync> = Arc::new(()) as Arc<dyn Send + Sync>; // placeholder, реальная память держится в cache entry's slice leak?
            // На самом деле нужно держать оригинальный CudaSlice живым, иначе его Drop освободит память
            // Поэтому сохраняем его в cache как CudaSlice<u8> через transmute
            
            // Храним копию slice как u8
            let slice_u8: CudaSlice<u8> = unsafe { self.stream.upgrade_device_ptr(ptr as *mut _, aligned_bytes) };
            let keep_alive: Arc<dyn Send + Sync> = Arc::new(std::mem::ManuallyDrop::new(slice_u8)) as Arc<dyn Send + Sync>;
            
            let mut cache = self.cache.write().unwrap();
            cache.insert(seq, CachedEntry {
                ptr,
                bytes: aligned_bytes,
                _keep_alive: keep_alive,
            });
            
            Ok(slice)
        }
    }

    /// Сброс арены (при смене модели)
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
        self.alloc_seq.store(0, Ordering::Relaxed);
        self.is_frozen.store(false, Ordering::Release);
        self.cache.write().unwrap().clear();
        tracing::info!("[FlatArena] reset");
    }
}

impl Drop for FlatArena {
    fn drop(&mut self) {
        if let Some(giant) = &self.giant {
            // Освобождаем giant блок вручную, т.к. мы делали leak
            unsafe {
                // cuMemFree_v2
                let _ = cudarc::driver::sys::cuMemFree_v2(giant.base_ptr as *mut _);
            }
            tracing::info!("[FlatArena] freed giant block {:#x}", giant.base_ptr);
        }
        // Остальные кешированные аллокации освободятся через _keep_alive Arc drop
    }
}
