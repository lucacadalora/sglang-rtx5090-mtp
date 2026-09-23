#!/usr/bin/env python3
"""WSL2 go/no-go for UVA reads of a pinned host table (the PR's unit tests, plus a full-size table and timing).
Run in a throwaway GPU container: bit-exact vs a device lookup, CUDA-graph capture, and gather speed."""
import time, torch, torch.nn.functional as F
from sglang.kernels.ops.embeddings.host_embedding_gather import host_embedding_gather

def check(ids, table):
    exp = F.embedding(ids.long(), table.cuda())
    got = host_embedding_gather(ids, table)
    torch.cuda.synchronize()
    assert torch.equal(got, exp), "MISMATCH"

for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for idt in (torch.int32, torch.int64):
        for h in (7, 128, 5120):
            t = torch.randn((64, h), dtype=dtype).pin_memory()
            check(torch.tensor([0, 63, 5, 5, 17], dtype=idt, device="cuda"), t)
print("small tables: bit-exact")

# full-size 27B table: 248,320 x 5,120 bf16 = 2.37 GiB pinned
big = torch.randn((248320, 5120), dtype=torch.bfloat16).pin_memory()
ids = torch.randint(0, 248320, (2048,), device="cuda")
got = host_embedding_gather(ids, big)
torch.cuda.synchronize()
ref = big[ids.cpu()].cuda()
assert torch.equal(got, ref), "MISMATCH on full-size table"
print("full-size 2.37 GiB table: bit-exact on 2,048 random rows")

# CUDA graph capture/replay
static = torch.zeros((9,), dtype=torch.int64, device="cuda")
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        host_embedding_gather(static, big)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = host_embedding_gather(static, big)
for _ in range(5):
    r = torch.randint(0, 248320, (9,), device="cuda")
    static.copy_(r)
    g.replay(); torch.cuda.synchronize()
    assert torch.equal(out, big[r.cpu()].cuda())
print("CUDA graph capture + replay: bit-exact")

def timeit(n):
    x = torch.randint(0, 248320, (n,), device="cuda")
    for _ in range(3):
        host_embedding_gather(x, big)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50):
        host_embedding_gather(x, big)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 * 1e6
for n in (1, 5, 64, 2048):
    print(f"gather {n:>5} rows: {timeit(n):8.1f} us")
print("WSL2 UVA GO")
