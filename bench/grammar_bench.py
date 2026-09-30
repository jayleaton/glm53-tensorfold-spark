#!/usr/bin/env python3
"""patches/0610: microbenchmark of the structured-output host work on GLM-5.3-Flash's real vocabulary (154,880
columns), CPU only.

Measures, for JSON (json_object, a nested schema) and a GLM tool call (the ``glm_4_7`` structural tag):

- the compilers' build (two tokenizer views) and each grammar's compile (cold, then cached);
- ``Constraint.cut`` (the walk that must precede the forward) per draft token;
- ``Constraint.fill`` per constrained row along real replies (p50 / p90 / max), and a whole 16-row window;
- a round of 4 slots x 16 rows: fills inline (one thread, GLM53_TF_GRAMMAR_THREADS=0) vs on 2 / 4 / 8 worker threads
  (the fill releases the GIL: the threads run beside the round's own thread);
- ``advance`` per token; ``apply`` on the host for [16, 77,440] (a rank's half; the GPU cost is in the GPU plan).

Usage (the tokenizer is not in the repo):
  GLM53_TF_TOKENIZER_DIR=<dir with tokenizer.json + config.json> PYTHONPATH=<tree>/src \\
      python bench/grammar_bench.py [--repeat N] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "year": {"type": "integer"}, "rating": {"type": "number"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "cast": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "role": {"type": "string"}}, "required": ["name", "role"],
        "additionalProperties": False}},
    "available": {"type": "boolean"}},
    "required": ["title", "year", "rating", "tags", "cast", "available"], "additionalProperties": False}
VALUE = {"title": "The Long Afternoon of Structured Output", "year": 2026, "rating": 8.25,
         "tags": ["drama", "speculative decoding", "grammar"],
         "cast": [{"name": "Ada Lovelace", "role": "Analyst"}, {"name": "Alan Turing", "role": "Machine"},
                  {"name": "Grace Hopper", "role": "Compiler"}], "available": True}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "strict": True, "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                     "units": {"type": "string", "enum": ["c", "f"]}},
    "required": ["city", "days", "units"], "additionalProperties": False}}}]
TOOL_TEXT = ["<tool_call>", "get_weather", "<arg_key>", "city", "</arg_key>", "<arg_value>", "Paris, France",
             "</arg_value>", "<arg_key>", "days", "</arg_key>", "<arg_value>", "5", "</arg_value>", "<arg_key>",
             "units", "</arg_key>", "<arg_value>", "c", "</arg_value>", "</tool_call>"]


def us(x: float) -> float:
    return round(x * 1e6, 1)


def ms(x: float) -> float:
    return round(x * 1e3, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    tokdir = Path(os.environ.get("GLM53_TF_TOKENIZER_DIR", ""))
    if not (tokdir / "tokenizer.json").is_file():
        print("set GLM53_TF_TOKENIZER_DIR to a directory with the checkpoint's tokenizer.json and config.json",
              file=sys.stderr)
        return 2
    import torch
    from tokenizers import Tokenizer

    from tensorfold.families.glm5_next.cuda import grammar

    out: dict = {"host": os.uname().machine, "cpus": os.cpu_count()}
    t0 = time.perf_counter()
    g = grammar.Grammars.from_model(tokdir, grammar.vocab_size(tokdir), (154820, 154827, 154829))
    out["build_s"] = round(time.perf_counter() - t0, 2)
    tok = Tokenizer.from_file(str(tokdir / "tokenizer.json"))
    marks = {s: tok.token_to_id(s) for s in grammar.TOOL_TOKENS}

    def ids(parts):
        r = []
        for p in parts:
            r += [marks[p]] if p in marks else tok.encode(p, add_special_tokens=False).ids
        return r

    cases = {
        "json_object": (grammar.Spec("json"), tok.encode(json.dumps(VALUE, indent=2), add_special_tokens=False).ids),
        "json_schema": (grammar.Spec("json_schema", json.dumps(SCHEMA)),
                        tok.encode(json.dumps(VALUE), add_special_tokens=False).ids),
        "tool_call": (grammar.request_spec({"tools": TOOLS, "tool_choice": "required"}), ids(TOOL_TEXT)),
    }
    for name, (spec, seq) in cases.items():
        res: dict = {"tokens": len(seq)}
        t0 = time.perf_counter()
        compiled = g.compile(spec)
        res["compile_ms_cold"] = ms(time.perf_counter() - t0)
        t0 = time.perf_counter()
        g.compile(spec)
        res["compile_ms_cached"] = ms(time.perf_counter() - t0)
        b = grammar.Bound(spec, compiled, g.think_end, True)
        # per row fill along the reply, and the cut walk per draft token
        fills, cuts, adv = [], [], []
        for _ in range(args.repeat):
            con = g.constraint(b)
            for i, t in enumerate(seq[:-1]):
                win = con.cut([seq[i - 1] if i else 0])
                t1 = time.perf_counter()
                con.fill(win)
                fills.append(time.perf_counter() - t1)
                window = [seq[i - 1] if i else 0] + seq[i:i + 15]
                t1 = time.perf_counter()
                w = con.cut(window)
                cuts.append((time.perf_counter() - t1) / max(1, len(window) - 1))
                t1 = time.perf_counter()
                con.advance([t])
                adv.append(time.perf_counter() - t1)
        fills.sort()
        res.update(fill_us_p50=us(statistics.median(fills)), fill_us_p90=us(fills[int(0.9 * len(fills))]),
                   fill_us_max=us(fills[-1]), fill_us_mean=us(statistics.fmean(fills)),
                   cut_us_per_draft=us(statistics.median(cuts)), advance_us=us(statistics.median(adv)))
        # a 16-row window at every 8th position: cut + fill (inline)
        wins = []
        con = g.constraint(b)
        for i in range(1, len(seq) - 16, 8):
            con.advance(seq[max(0, i - 8):i] if i > 1 else seq[:1])
            window = seq[i - 1:i + 15]
            t1 = time.perf_counter()
            con.fill(con.cut(window))
            wins.append(time.perf_counter() - t1)
        res["window16_ms"] = ms(statistics.median(wins)) if wins else None
        out[name] = res
    # a round of 4 slots x 16 rows (the slots at different points of the reply)
    for name in ("json_schema", "json_object"):
        spec, seq = cases[name]
        b = grammar.Bound(spec, g.compile(spec), g.think_end, True)
        starts = [1, len(seq) // 4, len(seq) // 2, 3 * len(seq) // 4]
        rounds = {}
        for n in (0, 2, 4, 8):
            filler = grammar.Filler(n)
            times = []
            for _ in range(args.repeat * 4):
                items = []
                for s in starts:
                    con = g.constraint(b)
                    con.advance(seq[:s])
                    items.append((con, con.cut(seq[s - 1:s + 15])))
                t1 = time.perf_counter()
                errs = filler.start(items).wait()
                times.append(time.perf_counter() - t1)
                assert errs == [None] * 4
            rounds[f"threads_{n}"] = ms(statistics.median(times))
        out[f"round_4x16_ms_{name}"] = rounds
    # apply on the host for a rank's half (16 rows, 77,440 columns)
    con = g.constraint(b)
    win = con.fill(con.cut(seq[:16]))
    x = torch.randn((16, 77440))
    t1 = time.perf_counter()
    for _ in range(20):
        grammar.apply(x, win, 77440)
    out["apply_cpu_ms_16x77440"] = ms((time.perf_counter() - t1) / 20)
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
