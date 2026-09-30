"""patches/0580: ``exl3_ld.cu`` compiled for sm_121 (GB10) WITHOUT a GPU, next to exl3.cu's grouped_kernel.

- Every ld_kernel instantiation the host dispatch can launch (and the probes), nvcc ``-arch=sm_121 -O3`` with torch's
  half / bf16 defines: no spills, registers within the launch bounds (NT 8: 3 CTAs an SM = 168 a thread; NT 4: 4 CTAs =
  128), static shared memory = grouped_kernel's (red + rows_sh: the staging area / ring lives inside red), resident
  CTAs an SM at least grouped_kernel's; the table (registers, CTAs an SM, nominal KB in flight an SM) is printed.
- The PTX has what the design names: mma.m16n8k16 f16 -> f32, ``ld.global.nc.L1::no_allocate.v4.u32`` (nc),
  ``ld.global.nc.L1::no_allocate.u32`` (w32), 16-byte ``cp.async.cg`` + ``cp.async.wait_group`` (cpa).
- The device helpers ld_kernel shares with exl3.cu (codebook, tile decode, mma) are character for character exl3.cu's.
- SASS (when a cuobjdump is found: $CUOBJDUMP, next to nvcc, or Triton's bundled one): inside the k loop of the default
  kernel (nc, 8, 2) the 128-bit trellis loads of step s + 2 sit between the tensor-core instructions of step s, i.e.
  the loads are in flight while the warp decodes and multiplies (grouped_kernel: every load of a step is issued, then
  waited for, before its first HMMA).

    NVCC=/usr/local/cuda/bin/nvcc PYTHONPATH=<patched tree>/src pytest -q -s tests/test_decode_loads_compile.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SMEM_MAX = 101376
SM_SMEM = 102400
REG_FILE = 65536
FLAGS = ["-arch=sm_121", "-O3", "-include", "cstdint", "-D__CUDA_NO_HALF_OPERATORS__",
         "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
         "--expt-relaxed-constexpr"]
SIG = "(const half*,const half*,const uint32_t*,const uint32_t*,const int*,const int*,const int*,float*,int,int,int,int,int,int,int)"
LDS = {"nc": 0, "w32": 1, "cpa": 2}
CFGS = [("nc", 8, 2), ("nc", 8, 1), ("nc", 8, 4), ("nc", 4, 2), ("nc", 4, 4), ("w32", 8, 2), ("w32", 4, 4),
        ("cpa", 8, 2), ("cpa", 4, 4)]
PROBES = [("nc", 1), ("nc", 2), ("nc", 3), ("w32", 3), ("cpa", 3)]


def _src_dir() -> Path:
    try:
        from tensorfold.families.glm5_next.cuda import __file__ as f
    except ImportError:
        pytest.skip("needs the patched tree on PYTHONPATH")
    d = Path(f).parent
    if not (d / "exl3_ld.cu").exists():
        pytest.skip("needs patches/0580")
    return d


def _nvcc() -> str:
    for c in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).exists():
            return c
    pytest.skip("no nvcc (set NVCC=...)")


def _cuobjdump() -> str | None:
    cands = [os.environ.get("CUOBJDUMP"), shutil.which("cuobjdump"), str(Path(_nvcc()).parent / "cuobjdump")]
    try:
        import triton

        cands.append(str(Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"))
    except ImportError:
        pass
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def _device_only(src: str, inst: list[str]) -> str:
    for h in ("#include <ATen/ATen.h>\n", "#include <ATen/cuda/CUDAContext.h>\n", "#include <c10/cuda/CUDAGuard.h>\n"):
        src = src.replace(h, "")
    src = src[:src.index("template <int NT, int PD, int LD, int PR>\nvoid launch(")].replace("namespace {", "", 1)
    return src + "\n" + "\n".join(inst) + "\n"


def _compile(src: str, td: Path, name: str):
    nvcc = _nvcc()
    cu = td / f"{name}.cu"
    cu.write_text(src)
    r = subprocess.run([nvcc, *FLAGS, "-cubin", "-Xptxas", "-v", "-o", str(td / f"{name}.cubin"), str(cu)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    p = subprocess.run([nvcc, *FLAGS, "-ptx", "-o", str(td / f"{name}.ptx"), str(cu)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    stats, fn = {}, None
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            fn = m.group(1)
            stats[fn] = {}
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and fn:
            stats[fn]["spill"] = int(m.group(1)) + int(m.group(2))
        m = re.search(r"Used (\d+) registers", line)
        if m and fn:
            stats[fn]["regs"] = int(m.group(1))
            s = re.search(r"(\d+) bytes smem", line)
            stats[fn]["smem"] = int(s.group(1)) if s else 0
    return stats, (td / f"{name}.ptx").read_text(), td / f"{name}.cubin"


def _ctas(regs: int, threads: int, smem: int) -> int:
    by_regs = REG_FILE // (((regs + 7) // 8) * 8 * threads)
    by_smem = SM_SMEM // (smem + 1024)
    return min(by_regs, by_smem, 1536 // threads)


@pytest.fixture(scope="module")
def built():
    d = _src_dir()
    inst = [f"template __global__ void ld_kernel<{nt},{pd},{LDS[ld]},0>{SIG};" for ld, nt, pd in CFGS]
    inst += [f"template __global__ void ld_kernel<8,2,{LDS[ld]},{p}>{SIG};" for ld, p in PROBES]
    ref_inst = [f"template __global__ void grouped_kernel<{nt},4>{SIG};" for nt in (8, 4)]
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        stats, ptx, cubin = _compile(_device_only((d / "exl3_ld.cu").read_text(), inst), td, "ld")
        ref_src = (d / "exl3.cu").read_text()
        ref_src = ref_src.replace("}  // namespace", "template <int NT, int PD, int LD, int PR>\nvoid launch(", 1)
        rstats, _, _ = _compile(_device_only(ref_src, ref_inst), td, "ref")
        sass = None
        cuobj = _cuobjdump()
        if cuobj:
            fn = next(n for n in stats if re.search(r"ld_kernelILi8ELi2ELi0ELi0E", n))
            r = subprocess.run([cuobj, "-sass", "-fun", fn, str(cubin)], capture_output=True, text=True)
            sass = r.stdout if r.returncode == 0 else None
        yield stats, ptx, rstats, sass, d


def test_compiles_and_fits(built):
    stats, _, rstats, _, _ = built
    assert len(stats) == len(CFGS) + len(PROBES)
    ref = {int(re.search(r"grouped_kernelILi(\d+)E", n).group(1)): st for n, st in rstats.items()
           if "grouped_kernel" in n}
    for nt, st in sorted(ref.items()):
        print(f"exl3.cu grouped_kernel<{nt},4>: {st['regs']} registers, {st['smem']} B shared -> "
              f"{_ctas(st['regs'], 128, st['smem'])} CTAs an SM")
    for name, st in sorted(stats.items()):
        nt, pd, ld, probe = map(int, re.search(r"ld_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", name).groups())
        ctas = _ctas(st["regs"], 128, st["smem"])
        kb = ctas * 4 * pd * nt * 128 / 1024
        ldn = [k for k, v in LDS.items() if v == ld][0]
        print(f"exl3_ld ld={ldn} NT={nt} PD={pd} probe={probe}: {st['regs']} registers, {st['spill']} B spilled, "
              f"{st['smem']} B shared -> {ctas} CTAs an SM, {kb:.0f} KB an SM in flight (nominal)")
        assert st["spill"] == 0
        assert st["regs"] <= (168 if nt == 8 else 128)          # __launch_bounds__(128, 3 | 4)
        assert st["smem"] == ref[nt]["smem"]                     # the staging area / ring is inside red
        assert st["smem"] <= SMEM_MAX
        assert ctas >= _ctas(ref[nt]["regs"], 128, ref[nt]["smem"]) or nt == 4 and ctas >= 4
        assert kb >= 4                                           # W16 littles' knee, with every warp resident


def test_ptx_has_the_design(built):
    _, ptx, _, _, _ = built
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    assert "ld.global.nc.L1::no_allocate.v4.u32" in ptx
    assert "ld.global.nc.L1::no_allocate.u32" in ptx
    assert re.search(r"cp\.async\.cg\.shared\.global \[[^]]+\], \[[^]]+\], 16;", ptx)
    assert "cp.async.wait_group" in ptx
    assert not re.search(r"(?<![\w.])(atom|red)\.", ptx)          # no atomics: sums in a fixed order only


def _functions(src: str, names: list[str]) -> dict[str, str]:
    out = {}
    for n in names:
        m = re.search(r"__device__ __forceinline__ [^\n]*\b" + n + r"\(", src)
        assert m, n
        i = src.index("{", m.start())
        depth = 0
        for j in range(i, len(src)):
            depth += src[j] == "{"
            depth -= src[j] == "}"
            if depth == 0:
                out[n] = src[m.start():j + 1]
                break
    return out


def test_pdl_prologue(built):
    """Before griddepcontrol.wait (the PDL prologue, first in program text): the grouping read with ld.global.cg, the
    trellis with ld.global.nc.L1::no_allocate or cp.async, nothing else -- no X / Xd load (ld.global.nc without
    no_allocate), no store to global memory (Z is written after the wait)."""

    _, ptx, _, _, _ = built
    bodies = re.split(r"\.entry ", ptx)[1:]
    assert bodies
    for body in bodies:
        name = body.split("(")[0]
        i = body.find("griddepcontrol.wait")
        assert i > 0, name
        pre = body[:i]
        assert not re.search(r"\bst\.global", pre), name
        for ld in re.findall(r"\bld\.global[.\w:]*", pre):
            assert ld.startswith("ld.global.cg") or ld.startswith("ld.global.nc.L1::no_allocate"), (name, ld)
        post = body[i:]
        assert re.search(r"ld\.global\.nc\.u32", post), name          # the X / Xd fragments come after it


def test_helpers_are_verbatim(built):
    d = built[4]
    mine, base = (d / "exl3_ld.cu").read_text(), (d / "exl3.cu").read_text()
    names = ["mcg2", "decode_tile", "mma16816"]
    a, b = _functions(mine, names), _functions(base, names)
    for n in names:
        assert a[n] == b[n], n
    # member_tiles' reduction (MG = 1): the same statements in the same order
    red = ["s = red[0][row][col];", "for (int w = 1; w < W; ++w) s += red[w][row][col];", "if (r < 0) continue;"]
    for line in red:
        assert line in mine and line in base, line


def test_sass_loads_in_flight_during_mma(built):
    sass = built[3]
    if not sass:
        pytest.skip("no cuobjdump (set CUOBJDUMP=...)")
    ops = [("LDG128" if re.search(r"LDG\.E\.[\w.]*128", ln) else "HMMA") for ln in sass.splitlines()
           if re.search(r"LDG\.E\.[\w.]*128|HMMA\.16816", ln)]
    # after the prologue's loads, some 128-bit loads must come after an HMMA and before a later HMMA of the same loop
    first_h = ops.index("HMMA")
    inside = [i for i in range(first_h, len(ops)) if ops[i] == "LDG128" and "HMMA" in ops[i + 1:]]
    print(f"SASS nc,8,2: {ops.count('LDG128')} LDG.128, {ops.count('HMMA')} HMMA, {len(inside)} loads between HMMAs")
    assert len(inside) >= 2
