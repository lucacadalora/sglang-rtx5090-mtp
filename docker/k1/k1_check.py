#!/usr/bin/env python3
"""k1 image build check: every .py file the k1 patches modified (newer than the stamp file) must byte-compile and
import, and every k1 env gate must be present. Usage: python3 k1_check.py <stamp-file>"""
import importlib
import os
import pathlib
import py_compile
import sys

PKG_ROOT = pathlib.Path("/sgl-workspace/sglang/python")
S = PKG_ROOT / "sglang"
stamp = os.path.getmtime(sys.argv[1])
touched = sorted(p for p in S.rglob("*.py") if p.stat().st_mtime > stamp)
assert touched, "no files modified by the k1 patches"
print("k1 touched files:")
for p in touched:
    print("  ", p.relative_to(PKG_ROOT))
    py_compile.compile(str(p), doraise=True)
for p in touched:
    mod = ".".join(p.relative_to(PKG_ROOT).with_suffix("").parts)
    importlib.import_module(mod)
    print("  import OK", mod)
text = "\n".join(p.read_text() for p in touched)
gates = [
    "SGLANG_GDN_BA_TINY_GEMM",  # p1
    "SGLANG_KV_SKIP_UNIT_SCALE_DIV",  # p2
    "SGLANG_AUTOTUNE_TARGET_LMHEAD",  # p3
    "SGLANG_LMHEAD_FORCE_TACTIC",  # p3
    "SGLANG_ATTN_MLP_SILU_FP4_FUSION",  # p7
    "SGLANG_EAGLE_SYNCFREE_SEAM",  # p6
]
missing = [g for g in gates if g not in text]
assert not missing, f"k1 env gates missing: {missing}"
print("k1 checks OK:", len(touched), "files,", len(gates), "env gates")
