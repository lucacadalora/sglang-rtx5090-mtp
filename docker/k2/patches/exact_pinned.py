"""k2: exact-size pinned + mapped host tensors from cudaHostAlloc (installed as sglang.srt.mem_cache.pool_host.exact_pinned).

Why this exists:
- On WSL2 (GPU paravirtualization) cudaDevAttrCanUseHostPointerForRegisteredMem is 0: memory pinned with
  mmap + cudaHostRegister (SGLang's default HiCache host pool) is reachable from the GPU only through a separate
  device alias. The installed sgl_kernel has no get_device_accessible_ptr, so HiCache kernels get raw host
  addresses and fault. Memory from cudaHostAlloc is mapped at the same address on host and device (measured on
  this box 2026-09-24: all six HiCache transfer paths byte-exact, 50 GB/s both directions).
- torch's pin_memory also uses cudaHostAlloc, but its caching host allocator rounds every block up to a power of
  two (the 2.37 GiB embedding table pins 4 GiB). This helper allocates exactly the bytes asked for.

The memory lives for the life of the process (host pools and the embedding table are never freed early).
"""
import ctypes
import logging
import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CUDA_HOST_ALLOC_PORTABLE = 0x01
_CUDA_HOST_ALLOC_MAPPED = 0x02
_lib = None
_lock = threading.Lock()


def _cudart():
    """The libcudart torch already loaded (same runtime instance as the rest of the process)."""
    global _lib
    with _lock:
        if _lib is None:
            torch.cuda.init()
            path = None
            with open("/proc/self/maps") as f:
                for line in f:
                    if "libcudart.so" in line:
                        path = line.split()[-1]
                        break
            lib = ctypes.CDLL(path or "libcudart.so.13")
            lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
            lib.cudaHostAlloc.restype = ctypes.c_int
            lib.cudaHostGetDevicePointer.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            lib.cudaHostGetDevicePointer.restype = ctypes.c_int
            lib.cudaGetErrorString.argtypes = [ctypes.c_int]
            lib.cudaGetErrorString.restype = ctypes.c_char_p
            _lib = lib
    return _lib


def exact_pinned_empty(dims, dtype: torch.dtype) -> torch.Tensor:
    """Uninitialised CPU tensor in pinned, device-mapped memory of exactly prod(dims) * itemsize bytes."""
    dims = tuple(int(d) for d in dims)
    nbytes = int(np.prod(dims, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size()
    if nbytes == 0:
        return torch.empty(dims, dtype=dtype)
    lib = _cudart()
    ptr = ctypes.c_void_p()
    rc = lib.cudaHostAlloc(ctypes.byref(ptr), nbytes, _CUDA_HOST_ALLOC_PORTABLE | _CUDA_HOST_ALLOC_MAPPED)
    if rc != 0:
        raise RuntimeError(
            f"cudaHostAlloc of {nbytes / 1e9:.2f} GB failed: rc={rc} ({lib.cudaGetErrorString(rc).decode()})"
        )
    alias = ctypes.c_void_p()
    rc = lib.cudaHostGetDevicePointer(ctypes.byref(alias), ptr, 0)
    if rc != 0 or alias.value != ptr.value:
        raise RuntimeError(
            "cudaHostAlloc memory is not mapped at the same address on the device "
            f"(rc={rc}); kernels that take raw host pointers would fault"
        )
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr.value)
    return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).view(dims)
