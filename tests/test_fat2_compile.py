"""patches/0590: ``exl3_fat2.cu`` compiled for sm_121 (GB10) WITHOUT a GPU, next to exl3_fast.cu's fat kernels.

- Resources of every instantiation the host can launch (3 configurations x gate/up, down, and the probes): 0 spills,
  <= 168 registers (__maxnreg__), static + dynamic shared memory <= 99 KB a CTA, one 256-thread CTA an SM, and what
  is left of the SM beside it for a CTA of patches/0084's overlap stream (the table is printed).
- The helpers are exl3_fast.cu's text (mcg2, decode_tile, mma16816, bf16r, the cp.async / ldmatrix wrappers, the plan
  kernel, fwht_row, swz), and the epilogue's arithmetic lines are fat's.
- PTX dataflow (``tests/fat2_ptx.py``): every value fat2 stores to global memory has the expression tree of a value fat
  stores (the same opcodes with their rounding / approximation modifiers and fma contractions -- expf's range reduction,
  the ``1 + e`` folded into an fma, div.rn, the bf16 / fp16 conversions, the shuffles and selects of both transforms --
  the same constants and operand order): gate/up against fat's gate/up, down against fat's down; the trees reach
  ex2 / div.rn / cvt.rn.bf16 / shfl.bfly and never an undefined register.
- PTX mma: one instruction form (m16n8k16.row.col.f32.f16.f16.f32), C == D registers (one accumulator chain an mma),
  A = the decode tree of one shared-memory word, B = ldmatrix registers (0, 1) / (2, 3), the same set as fat's.
- PTX atomics: only the ticket (atom.global.add) and the member count (atom.shared.max); no red.
- SASS (when a cuobjdump is found): the floating-point instruction census of the epilogue (FADD / FMUL / FFMA with
  their modifiers, MUFU.EX2, F2F, FMNMX, FSEL, FCHK, HADD2.F32): fat2 with 64 members (2 epilogue rounds) == fat's
  gate/up and down-large (2 rounds); 128 members (4 rounds) adds exactly two rounds, fat's own per-round delta (so ptxas
  contracted nothing differently at fat2's register cap); no local memory; at least the unrolled loop's HMMA.
- Negative controls: three planted source changes (an extra rounding in the gate scale, an extra multiply before the
  down transform, the two B registers of an mma swapped) are caught by the PTX checks.

    NVCC=/usr/local/cuda/bin/nvcc TF_SRC=<patched tree>/src pytest -q -s tests/test_fat2_compile.py   (~1 minute)
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import fat2_ptx as P  # noqa: E402

SMEM_MAX = 101376                  # a CTA's shared memory on GB10
SM_SMEM = 102400                   # an SM's
CTA_RESERVED = 1024                # the runtime's reserve a CTA
REG_FILE = 65536
REGS_CAP = 168
EPI = 2 * 32 * 132 * 4


def _src_dir() -> Path:
    env = os.environ.get("TF_SRC")
    if env:
        d = Path(env) / "tensorfold" / "families" / "glm5_next" / "cuda"
    else:
        try:
            from tensorfold.families.glm5_next.cuda import __file__ as f
        except ImportError:
            pytest.skip("needs the patched tree (TF_SRC=<tree>/src or on PYTHONPATH)")
        d = Path(f).parent
    if not (d / "exl3_fat2.cu").exists():
        pytest.skip("needs patches/0590")
    return d


def dyn_smem(ng: int, nsa: int) -> int:
    a = ng * 8 * 32 * 2
    w = 2 * 2 * 8 * 32 * 4
    return nsa * (a + w) + EPI


@pytest.fixture(scope="module")
def built():
    if P.nvcc() is None:
        pytest.skip("no nvcc (set NVCC=...)")
    d = _src_dir()
    work = Path(tempfile.mkdtemp(prefix="fat2-"))
    fat = P.compile_ptx(P.fat_device_source((d / "exl3_fast.cu").read_text()), work, "fat")
    fat2 = P.compile_ptx(P.fat2_device_source((d / "exl3_fat2.cu").read_text()), work, "fat2")
    return {"dir": d, "work": work, "fat": fat, "fat2": fat2}


def test_resources(built):
    res = P.resources(built["fat2"][1])
    fres = P.resources(built["fat"][1])
    print("\n  kernel (kind, members, stages, probe)   regs spill  smem KB  CTAs/SM  left for a side CTA")
    for k in P.FAT2_KERNELS:
        r = res[P.mangled_fat2(*k)]
        total = r["smem"] + dyn_smem(k[1], k[2])
        per_sm = min(REG_FILE // (r["regs"] * 256), SM_SMEM // (total + CTA_RESERVED))
        left_regs = REG_FILE - r["regs"] * 256
        left_smem = SM_SMEM - (total + CTA_RESERVED) - CTA_RESERVED
        print(f"  {str(k):40s} {r['regs']:4d} {r['spill']:5d} {total / 1024:8.1f} {per_sm:8d}  "
              f"{left_regs:,} registers, {left_smem / 1024:.1f} KB")
        assert r["spill"] == 0, (k, r)
        assert r["regs"] <= REGS_CAP, (k, r)
        assert total <= SMEM_MAX, (k, total)
        assert per_sm >= 1
    for key in P.FAT_KERNELS:
        r = fres[P.mangled_fat(key)]
        print(f"  fat {key:36s} {r['regs']:4d} {r['spill']:5d}   (fat: 128-256 threads, 1-2 CTAs an SM)")
    # the default configuration leaves room beside it for a 256-thread CTA at <= 80 registers and ~16 KB
    r = res[P.mangled_fat2(0, 16, 4, 0)]
    assert REG_FILE - r["regs"] * 256 >= 256 * 80
    assert SM_SMEM - (r["smem"] + dyn_smem(16, 4) + CTA_RESERVED) - CTA_RESERVED >= 15 * 1024


# -- source text ---------------------------------------------------------------------------------------------------------
def _body(text: str, signature: str) -> str:
    i = text.index(signature)
    j = text.index("{", i)
    depth, k = 0, j
    while True:
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[i:k + 1]
        k += 1


HELPERS = ["__device__ __forceinline__ uint32_t mcg2(", "__device__ __forceinline__ void decode_tile(",
           "__device__ __forceinline__ void mma16816(", "__device__ __forceinline__ float bf16r(",
           "__device__ __forceinline__ void cp_async16(", "__device__ __forceinline__ void cp_commit(",
           "__device__ __forceinline__ void cp_wait(", "__device__ __forceinline__ void ldsm4(",
           "__global__ void plan_kernel(", "__device__ __forceinline__ void fwht_row(",
           "__device__ __forceinline__ int swz("]


@pytest.mark.parametrize("sig", HELPERS)
def test_helpers_verbatim(sig):
    d = _src_dir()
    a = (d / "exl3_fast.cu").read_text()
    b = (d / "exl3_fat2.cu").read_text()
    assert _body(a, sig) == _body(b, sig)


def _arith(text: str) -> collections.Counter:
    keys = ("HAD_SCALE", "bf16r(", "expf(", "fwht_row(", "__float2half_rn", "__half2float(sv[", "float4 o;")
    out = collections.Counter()
    for line in text.splitlines():
        s = line.strip()
        if any(k in s for k in keys) and not s.startswith("//") and "constexpr" not in s and "__device__" not in s:
            out[re.sub(r"\s+", " ", s)] += 1
    return out


def test_epilogue_arithmetic_lines_are_fats():
    d = _src_dir()
    fast = (d / "exl3_fast.cu").read_text()
    fat = fast[fast.index("namespace fat {"):fast.index("// rot_in for one matrix")]
    fat_epi = _arith(fat[fat.index("// epilogue: fast2's, unchanged"):])
    src2 = (d / "exl3_fat2.cu").read_text()
    k2 = src2[src2.index("template <int KIND, int NG, int NSA, int PROBE>\n__global__"):src2.index("void launch(")]
    fat2_epi = _arith(k2[k2.index("// epilogue: fat's"):])
    assert set(fat2_epi) == set(fat_epi), (set(fat2_epi) ^ set(fat_epi))


# -- PTX -------------------------------------------------------------------------------------------------------------
def _pairs():
    """(fat kernel, fat2 kernels with the same epilogue kind)."""

    gu = [k for k in P.FAT2_KERNELS if k[0] == 0]
    dn = [k for k in P.FAT2_KERNELS if k[0] == 1]
    return [("gu", gu), ("dn", dn), ("dn_large", dn)]


def test_ptx_stored_values_are_fats(built):
    fptx, p2 = built["fat"][0], built["fat2"][0]
    for key, kernels in _pairs():
        want = set(P.stored_trees(fptx, P.mangled_fat(key)))
        assert want
        for k in kernels:
            got = set(P.stored_trees(p2, P.mangled_fat2(*k)))
            assert got == want, (key, k)


def test_ptx_trees_reach_the_arithmetic(built):
    fptx, p2 = built["fat"][0], built["fat2"][0]
    need_gu = {"ex2.approx.ftz.f32", "div.rn.f32", "cvt.rn.bf16.f32", "cvt.rn.f16.f32", "shfl.sync.bfly.b32",
               "selp.f32", "fma.rn.f32", "min.f32", "max.f32", "ld.shared.v4.b32", "ld.global.nc.b16"}
    need_dn = {"shfl.sync.bfly.b32", "selp.f32", "mul.f32", "ld.shared.v4.b32", "ld.global.nc.b16"}
    for ptx, name, need in ((fptx, P.mangled_fat("gu"), need_gu), (p2, P.mangled_fat2(0, 16, 4, 0), need_gu),
                            (fptx, P.mangled_fat("dn"), need_dn), (p2, P.mangled_fat2(1, 16, 4, 0), need_dn)):
        ops = P.reached_ops(ptx, name)
        assert need <= ops and "UNDEF" not in ops, (name, need - ops)


def test_ptx_mma_operands_are_fats(built):
    fptx, p2 = built["fat"][0], built["fat2"][0]
    for key, kernels in _pairs():
        f = P.mma_operands(fptx, P.mangled_fat(key))
        assert f and all(c for _, _, c in f)
        want = {(a, b) for a, b, _ in f}
        for k in kernels:
            if k[3]:
                continue
            got = P.mma_operands(p2, P.mangled_fat2(*k))
            assert all(c for _, _, c in got), k
            assert {(a, b) for a, b, _ in got} == want, (key, k)
            assert P.mma_lines(p2, P.mangled_fat2(*k)) == P.mma_lines(fptx, P.mangled_fat(key)) == \
                {"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"}


def test_ptx_atomics_are_the_ticket_and_the_count(built):
    p2 = built["fat2"][0]
    for k in P.FAT2_KERNELS:
        ops = P.opcodes(p2, P.mangled_fat2(*k))
        atoms = {o for o in ops if o.startswith(("atom.", "red."))}
        assert atoms <= {"atom.global.add.u32", "atom.shared.max.s32"}, (k, atoms)
        assert "atom.global.add.u32" in atoms and "atom.shared.max.s32" in atoms, (k, atoms)


# -- SASS --------------------------------------------------------------------------------------------------------------
FP_OPS = re.compile(r"^(FADD|FMUL|FFMA|FMNMX|FSEL|FCHK|MUFU\.EX2|F2F\.BF16|F2FP\.F16|HADD2\.F32)")


def _sass(built, which: str) -> dict[str, str]:
    co = P.cuobjdump()
    if co is None:
        pytest.skip("no cuobjdump")
    text = subprocess.run([co, "-sass", str(built[which][2])], check=True, capture_output=True, text=True).stdout
    parts = re.split(r"\n\s*Function : (\S+)\n", text)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def _census(body: str) -> collections.Counter:
    c = collections.Counter()
    for ln in body.splitlines():
        m = re.match(r"^\s*/\*[0-9a-f]+\*/\s+(@!?U?P\w+\s+)?([A-Z][\w.]*)", ln)
        if m:
            c[m.group(2)] += 1
    return c


def _fp(c: collections.Counter) -> collections.Counter:
    return collections.Counter({k: v for k, v in c.items() if FP_OPS.match(k)})


def test_sass_epilogue_census_is_fats(built):
    f, f2 = _sass(built, "fat"), _sass(built, "fat2")
    fat = {k: _census(f[P.mangled_fat(k)]) for k in P.FAT_KERNELS}
    two = {k: _census(f2[P.mangled_fat2(*k)]) for k in P.FAT2_KERNELS}
    print()
    for name, c in list(fat.items()) + list(two.items()):
        print(f"  {str(name):18s} HMMA {c['HMMA.16816.F32']:4d}  " + " ".join(f"{k} {v}" for k, v in sorted(_fp(c).items())))
    # 2 epilogue rounds each: fat gate/up == fat2 gate/up with 64 members; fat down-large == fat2 down with 64
    assert _fp(two[(0, 8, 4, 0)]) == _fp(fat["gu"])
    assert _fp(two[(1, 8, 4, 0)]) == _fp(fat["dn_large"])
    # 4 rounds: exactly two more of fat's rounds (down: fat's dn-large minus dn is one round)
    per_round_dn = _fp(fat["dn_large"])
    per_round_dn.subtract(_fp(fat["dn"]))
    for k in ((1, 16, 4, 0), (1, 16, 3, 0)):
        want = _fp(two[(1, 8, 4, 0)])
        want.update({o: 2 * v for o, v in per_round_dn.items()})
        assert +_fp(two[k]) == +want, k
    gu_round = _fp(two[(0, 16, 4, 0)])
    gu_round.subtract(_fp(two[(0, 8, 4, 0)]))
    assert all(v % 2 == 0 and v >= 0 for v in gu_round.values())
    assert _fp(two[(0, 16, 3, 0)]) == _fp(two[(0, 16, 4, 0)])
    for k in P.FAT2_KERNELS:
        c = two[k]
        assert c["LDL"] == 0 and c["STL"] == 0, k
        if k[3] != 2:
            assert c["HMMA.16816.F32"] >= 2 * 2 * (k[1] // 2) * 2, k      # KS x MTL x NG / 2 x 2 (ptxas may peel)


# -- negative controls: planted changes the PTX checks must catch ----------------------------------------------------------
PLANTS = {
    "gate_scale": ("const float gg = fminf(bf16r(v[j] * HAD_SCALE * __half2float(sg[j])), limit);",
                   "const float gg = fminf(bf16r(v[j] * (HAD_SCALE * __half2float(sg[j]))), limit);", 0),
    "down_scale": ("o.x = v[0] * HAD_SCALE * __half2float(sv[0]);",
                   "o.x = v[0] * HAD_SCALE * __half2float(sv[0]) * 1.0000001f;", 1),
    "b_swap": ("mma16816(acc[l][2 * np], af[l], b[0], b[1]);", "mma16816(acc[l][2 * np], af[l], b[1], b[0]);", 0),
}


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_planted_changes_are_caught(built, plant):
    old, new, kind = PLANTS[plant]
    src = (built["dir"] / "exl3_fat2.cu").read_text()
    assert src.count(old) == 1
    ptx, _, _ = P.compile_ptx(P.fat2_device_source(src.replace(old, new), [(kind, 16, 4, 0)]), built["work"],
                              f"plant_{plant}")
    name = P.mangled_fat2(kind, 16, 4, 0)
    fptx = built["fat"][0]
    fkey = "gu" if kind == 0 else "dn"
    stores_same = set(P.stored_trees(ptx, name)) == set(P.stored_trees(fptx, P.mangled_fat(fkey)))
    mma_same = {(a, b) for a, b, _ in P.mma_operands(ptx, name)} == \
        {(a, b) for a, b, _ in P.mma_operands(fptx, P.mangled_fat(fkey))}
    assert not (stores_same and mma_same), plant
