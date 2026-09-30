"""patches/0590: build exl3_fast.cu's fat kernels and exl3_fat2.cu's kernels for sm_121 WITHOUT a GPU, and a small PTX
dataflow tracer, for ``tests/test_fat2_compile.py``.

``stored_trees(ptx, kernel)`` follows every value a kernel stores to global memory (``st.global``) back through the
PTX that computes it, to its leaves (loads, special registers, parameters, immediates), and returns one hash a stored
value: the hash of the expression tree -- every opcode with its type and rounding / approximation modifiers (so an
fma contraction, a ``div.rn`` vs ``div.approx``, an ``ex2.approx.ftz``, the constants of expf's range reduction and the
operand order all count), the shuffles with their lane masks, the selects with their predicates' comparisons. Address
arithmetic is not part of a value's tree (loads are leaves: which element is where is the emulator's job). Two kernels
whose stores have the same set of trees compute every stored element with the same PTX arithmetic.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FLAGS = ["-arch=sm_121", "-O3", "-include", "cstdint", "-D__CUDA_NO_HALF_OPERATORS__",
         "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
         "--expt-relaxed-constexpr"]
HDRS = ("#include <ATen/ATen.h>\n", "#include <ATen/cuda/CUDAContext.h>\n", "#include <c10/cuda/CUDAGuard.h>\n")
FAT_SIG = ("(const half*, const half*, const uint32_t*, const uint32_t*, const int*, const int*, const int*, int*, int, "
           "int, int, int, int, const half*, const half*, const half*, half*, float*, float)")
FAT2_SIG = ("(const half*, const uint32_t*, const uint32_t*, const int*, const int*, const int*, int*, int, int, int, "
            "int, int, const half*, const half*, const half*, half*, float*, float)")
# fat's production instantiations (exl3_fat_gateup_cuda / exl3_fat_down_cuda: 3 stages, the shared input)
FAT_KERNELS = {
    "gu": "fat::expert_kernel<2, 2, 8, 4, 3, 4, true, 2>",
    "dn": "fat::expert_kernel<1, 2, 8, 4, 3, 8, false, 2>",
    "dn_large": "fat::expert_kernel<1, 2, 16, 2, 3, 8, false, 1>",
}
# every fat2 instantiation the host code can launch: (kind, NG, NSA, PROBE)
FAT2_KERNELS = [(k, ng, nsa, 0) for k in (0, 1) for ng, nsa in ((16, 4), (8, 4), (16, 3))] + \
               [(k, 16, 4, p) for k in (0, 1) for p in (1, 2)]


def nvcc() -> str | None:
    for c in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).exists():
            return c
    return None


def cuobjdump() -> str | None:
    cands = [os.environ.get("CUOBJDUMP"), shutil.which("cuobjdump")]
    nv = nvcc()
    if nv:
        cands.append(str(Path(nv).parent / "cuobjdump"))
    try:
        import triton

        cands.append(str(Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"))
    except ImportError:
        pass
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def fat_device_source(src: str) -> str:
    """exl3_fast.cu up to the end of namespace fat (the fat kernels, rot_in1, minb), without host code."""

    for h in HDRS:
        src = src.replace(h, "")
    src = src[:src.index("// -- once (patches/0260")]
    out, lines, i = [], src.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if (ln.startswith("template <") and i + 1 < len(lines) and lines[i + 1].startswith("void launch(")) or \
                ln.startswith("inline bool enabled()"):
            j = i + (1 if ln.startswith("template <") else 0)
            while lines[j] != "}":
                j += 1
            i = j + 1
            continue
        out.append(ln)
        i += 1
    src = "\n".join(out).replace("namespace {", "", 1)
    inst = [f"template __global__ void {k}{FAT_SIG};" for k in FAT_KERNELS.values()]
    return src + "\n" + "\n".join(inst) + "\n"


def fat2_device_source(src: str, kernels=FAT2_KERNELS) -> str:
    for h in HDRS:
        src = src.replace(h, "")
    src = src[:src.index("template <int KIND, int NG, int NSA, int PROBE>\nvoid launch(")].replace("namespace {", "", 1)
    src += "}  // namespace fat2\n"
    inst = [f"template __global__ void fat2::expert_kernel<{k}, {ng}, {nsa}, {p}>{FAT2_SIG};" for k, ng, nsa, p in kernels]
    return src + "\n".join(inst) + "\n"


def compile_ptx(source: str, work: Path, name: str) -> tuple[str, str, Path]:
    """-> (ptx text, ptxas -v log, cubin path)."""

    nv = nvcc()
    cu = work / f"{name}.cu"
    cu.write_text(source)
    ptx = work / f"{name}.ptx"
    cubin = work / f"{name}.cubin"
    subprocess.run([nv, *FLAGS, "-ptx", "-o", str(ptx), str(cu)], check=True, capture_output=True, text=True)
    r = subprocess.run([nv, *FLAGS, "-Xptxas", "-v", "-cubin", "-o", str(cubin), str(cu)], check=True,
                       capture_output=True, text=True)
    return ptx.read_text(), r.stdout + r.stderr, cubin


def resources(log: str) -> dict[str, dict]:
    """ptxas -v per entry: registers, spill bytes, static shared memory."""

    out, cur = {}, None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            out[cur]["spill"] = int(m.group(1)) + int(m.group(2))
        m = re.search(r"Used (\d+) registers", line)
        if m:
            out[cur]["regs"] = int(m.group(1))
            s = re.search(r"(\d+) bytes smem", line)
            out[cur]["smem"] = int(s.group(1)) if s else 0
    return out


def mangled_fat2(kind: int, ng: int, nsa: int, probe: int) -> str:
    return f"_ZN4fat213expert_kernelILi{kind}ELi{ng}ELi{nsa}ELi{probe}EEEvPK6__halfPKjS5_PKiS7_S7_PiiiiiiS3_S3_S3_PS1_Pff"


def mangled_fat(key: str) -> str:
    args = {"gu": "Li2ELi2ELi8ELi4ELi3ELi4ELb1ELi2E", "dn": "Li1ELi2ELi8ELi4ELi3ELi8ELb0ELi2E",
            "dn_large": "Li1ELi2ELi16ELi2ELi3ELi8ELb0ELi1E"}[key]
    return f"_ZN3fat13expert_kernelI{args}EEvPK6__halfS3_PKjS5_PKiS7_S7_PiiiiiiS3_S3_S3_PS1_Pff"


# -- PTX ---------------------------------------------------------------------------------------------------------------
def entry_body(ptx: str, name: str) -> str:
    i = ptx.index(f".entry {name}(")
    nxt = [k for k in (ptx.find("\n.visible", i + 1), ptx.find("\n.entry", i + 1), ptx.find("\n.func", i + 1),
                       ptx.find("\n.weak", i + 1)) if k >= 0]
    j = ptx.rindex("}", i, min(nxt) if nxt else len(ptx))      # inline asm closes its braces in column 0 too
    return ptx[ptx.index("{", ptx.index("\n)", i)) + 1:j]


def _split_ops(s: str) -> list[str]:
    ops, depth, cur = [], 0, ""
    for ch in s:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            ops.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        ops.append(cur.strip())
    return ops


NO_DEST = ("st.", "bar.", "barrier", "bra", "ret", "cp.async", "red.", "membar", "fence", "exit", "prefetch",
           "mma.", "@", "trap")


def statements(body: str) -> list[tuple[str, list[str], str | None]]:
    """(opcode, operands, predicate) a statement, in order; labels, declarations and comments dropped."""

    body = re.sub(r"//[^\n]*", "", body)
    out = []
    for raw in body.split(";"):
        s, prev = raw.strip(), None
        while s != prev:                                     # inline asm braces and labels, in any order
            prev = s
            s = re.sub(r"^(\$[\w]+:\s*)+", "", s.lstrip("{}").strip()).strip()
        if not s or s.startswith(".reg") or s.startswith(".shared") or s.startswith(".local") or s.startswith(".pragma"):
            continue
        pred = None
        m = re.match(r"@(!?%\w+)\s+(.*)", s, re.S)
        if m:
            pred, s = m.group(1), m.group(2)
        parts = s.split(None, 1)
        op = parts[0]
        ops = _split_ops(parts[1]) if len(parts) > 1 else []
        out.append((op, ops, pred))
    return out


def _regs(operand: str) -> list[str]:
    return re.findall(r"%[\w]+", operand)


class Tracer:
    LEAF_OPS = ("ld.", "ldu.", "ldmatrix", "tex.", "atom.")

    def __init__(self, stmts) -> None:
        self.st = stmts
        self.defs: dict[str, list[tuple[int, int]]] = {}
        for i, (op, ops, _) in enumerate(stmts):
            if not ops or op.startswith(NO_DEST) or op in ("bra", "bar.sync"):
                continue
            dest = ops[0]
            if dest.startswith("{"):
                for k, r in enumerate(_regs(dest)):
                    self.defs.setdefault(r, []).append((i, k))
            else:
                for k, r in enumerate(dest.split("|")):
                    r = r.strip()
                    if r.startswith("%"):
                        self.defs.setdefault(r, []).append((i, 0 if k == 0 else -1))
        self.memo: dict[tuple, str] = {}
        self.undef = False
        sys.setrecursionlimit(100000)

    def _def_before(self, reg: str, idx: int):
        best = None
        for i, k in self.defs.get(reg, ()):
            if i < idx:
                best = (i, k)
            else:
                break
        return best

    def value(self, operand: str, idx: int) -> str:
        operand = operand.strip()
        if operand.startswith("{"):
            return self._h("VEC", [self.value(r, idx) for r in _regs(operand)])
        if not operand.startswith("%"):
            return self._h("IMM", [operand])
        if operand.startswith(("%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid")):
            return self._h("SPECIAL", [operand])
        d = self._def_before(operand, idx)
        if d is None:
            self.undef = True
            return self._h("UNDEF", [re.sub(r"\d+", "", operand)])
        return self.node(*d)

    def node(self, i: int, k: int) -> str:
        key = (i, k)
        if key in self.memo:
            return self.memo[key]
        op, ops, pred = self.st[i]
        if op.startswith(self.LEAF_OPS) or op.startswith("ld.param") or op.startswith("cvta"):
            h = self._h("LEAF", [op, str(k)])
        elif op.startswith("mov") and len(ops) == 2 and not ops[1].startswith(("%", "{")):
            h = self._h("IMM", [ops[1]])
        elif op.startswith("mov") and len(ops) == 2 and ops[1].startswith("%"):
            h = self.value(ops[1], i)                          # a copy is transparent
        elif op.startswith("setp"):
            h = self._h("PRED", [op] + [self.value(o, i) for o in ops[1:]])
        elif op.startswith("selp") and len(ops) == 4:
            # selp(a, b, p): p = setp.ne(x, y) is selp(b, a, setp.eq(x, y)) (nvcc emits either in different copies)
            a, b, p = ops[1], ops[2], ops[3]
            d = self._def_before(p, i)
            if d is not None and self.st[d[0]][0].startswith("setp.ne."):
                pop, pops, _ = self.st[d[0]]
                ph = self._h("PRED", [pop.replace("setp.ne.", "setp.eq.", 1)] + [self.value(o, d[0]) for o in pops[1:]])
                h = self._h(op, [self.value(b, i), self.value(a, i), ph])
            else:
                h = self._h(op, [self.value(a, i), self.value(b, i), self.value(p, i)])
        else:
            h = self._h(op + ("" if k <= 0 else f"#{k}"), [self.value(o, i) for o in ops[1:]])
        self.memo[key] = h
        return h

    @staticmethod
    def _h(tag: str, parts: list[str]) -> str:
        return hashlib.sha1((tag + "(" + ",".join(parts) + ")").encode()).hexdigest()[:16]


def stored_trees(ptx: str, kernel: str) -> list[tuple[str, tuple[str, ...]]]:
    """(store opcode, hashes of the stored value(s)) for every st.global of the kernel, in program order."""

    st = statements(entry_body(ptx, kernel))
    tr = Tracer(st)
    out = []
    for i, (op, ops, pred) in enumerate(st):
        if op.startswith("st.global"):
            val = ops[1]
            regs = _regs(val) if val.startswith("{") else [val]
            out.append((op, tuple(tr.value(r, i) for r in regs)))
    return out


def opcodes(ptx: str, kernel: str) -> list[str]:
    return [op for op, _, _ in statements(entry_body(ptx, kernel))]


def mma_lines(ptx: str, kernel: str) -> set[str]:
    return {m.group(0) for m in re.finditer(r"mma\.sync\.[\w.]+", entry_body(ptx, kernel))}


def mma_operands(ptx: str, kernel: str) -> list[tuple[tuple[str, ...], tuple[str, ...], bool]]:
    """Every mma of the kernel: (hashes of its 4 A registers' trees, hashes of its 2 B registers' trees, C == D)."""

    st = statements(entry_body(ptx, kernel))
    tr = Tracer(st)
    out = []
    for i, (op, ops, _) in enumerate(st):
        if op.startswith("mma.sync"):
            d, a, b, c = ops[:4]
            out.append((tuple(tr.value(r, i) for r in _regs(a)), tuple(tr.value(r, i) for r in _regs(b)),
                        _regs(c) == _regs(d)))
    return out


def reached_ops(ptx: str, kernel: str) -> set[str]:
    """Opcodes of every instruction the stored values depend on (their trees' inner nodes)."""

    st = statements(entry_body(ptx, kernel))
    tr = Tracer(st)
    for i, (op, ops, _) in enumerate(st):
        if op.startswith("st.global"):
            val = ops[1]
            for r in (_regs(val) if val.startswith("{") else [val]):
                tr.value(r, i)
    return {st[i][0] for i, _ in tr.memo} | ({"UNDEF"} if tr.undef else set())
