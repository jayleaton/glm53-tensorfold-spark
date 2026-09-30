#!/usr/bin/env python3
"""patches/0610 GPU gates (docs/STRUCTURED-OUTPUT.md section 8): structured output against a running server.

Suites (``--suites``, comma list; all by default):

- ``schemas``: 8 JSON schemas (nested objects, arrays, enums, bounds, numbers) x greedy / seeded sampling x thinking
  on / off, each sent drafted alone, with ``"draft": false`` alone, and the 8 of a setting 4 at a time concurrently
  (4 slots). Gates: every variant of a request has the same token hash (``tensorfold.sha256``): drafted == serial ==
  batched; every reply that finished (``finish_reason`` stop) parses and validates (jsonschema when installed); the
  reasoning (thinking on) precedes the JSON and never leaks into it.
- ``rigmark``: RigMark's structured decode task shape (a 50-object JSON array, ``temperature`` 0, ``max_tokens``
  4096, thinking on), without and with ``response_format`` json_schema: both valid JSON of 50 objects; decode tok/s of
  each (``completion_tokens`` / ``decode_s``) and the grammar's stats (drafts cut, rows masked, fill / exposed wait
  ms) for the speed impact.
- ``tools``: ``tool_choice`` required, a named function and a strict tool (auto): every call parses into a known
  tool whose arguments validate against its schema, and no GLM markup leaks into argument strings.
- ``plain``: unconstrained requests (the exact prompts of ``--ref``, or three built-in ones): the token hashes of a
  knob-off run (``--ref`` file written by this suite on the knob-off server) are unchanged with the knob on.

Usage:
  python3 bench/structured.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --out R/structured.json \\
      [--suites schemas,rigmark,tools,plain] [--ref R/plain-off.json] [--write-ref R/plain-off.json]
Exit status 0 when every gate passes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from typing import Any

SCHEMAS: list[dict] = [
    {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0,
                                                                            "maximum": 130},
                                      "email": {"type": "string"}}, "required": ["name", "age", "email"],
     "additionalProperties": False},
    {"type": "object", "properties": {"title": {"type": "string"}, "tags": {"type": "array", "items": {
        "type": "string"}, "maxItems": 5}, "rating": {"type": "number"}}, "required": ["title", "tags", "rating"],
     "additionalProperties": False},
    {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok", "degraded", "down"]},
                                      "latency_ms": {"type": "integer"}, "checked": {"type": "boolean"}},
     "required": ["status", "latency_ms", "checked"], "additionalProperties": False},
    {"type": "object", "properties": {"order": {"type": "object", "properties": {
        "id": {"type": "string"}, "items": {"type": "array", "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}}, "required": ["sku", "qty"],
            "additionalProperties": False}, "maxItems": 4}}, "required": ["id", "items"],
        "additionalProperties": False}}, "required": ["order"], "additionalProperties": False},
    {"type": "object", "properties": {"city": {"type": "string"}, "forecast": {"type": "array", "items": {
        "type": "object", "properties": {"day": {"type": "string"}, "high_c": {"type": "number"},
                                         "low_c": {"type": "number"}}, "required": ["day", "high_c", "low_c"],
        "additionalProperties": False}, "maxItems": 3}}, "required": ["city", "forecast"],
     "additionalProperties": False},
    {"type": "object", "properties": {"summary": {"type": "string", "maxLength": 200},
                                      "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                                      "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
     "required": ["summary", "sentiment", "confidence"], "additionalProperties": False},
    {"type": "object", "properties": {"function": {"type": "string"}, "args": {"type": "object", "properties": {
        "path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"],
        "additionalProperties": False}}, "required": ["function", "args"], "additionalProperties": False},
    {"type": "object", "properties": {"matrix": {"type": "array", "items": {"type": "array", "items": {
        "type": "integer"}, "maxItems": 3}, "maxItems": 3}, "trace": {"type": "integer"}},
     "required": ["matrix", "trace"], "additionalProperties": False},
]
PROMPTS = [
    "Invent a person and give their name, age and email.",
    "Describe a science fiction novel you would like to read: its title, a few tags and a rating out of 10.",
    "Report the status of a web service you just checked.",
    "Create a small order for a hardware store with a few items.",
    "Give a three-day weather forecast for Lisbon in October.",
    "Summarize the sentiment of this review: 'The battery lasts forever but the screen scratches easily.'",
    "Which function would you call to list the files of /var/log, with which arguments?",
    "Write a 3x3 integer matrix of your choice and its trace.",
]
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                      "parameters": {"type": "object", "properties": {
                                          "city": {"type": "string"}, "units": {"type": "string",
                                                                                "enum": ["c", "f"]}},
                                          "required": ["city", "units"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
                                      "parameters": {"type": "object", "properties": {
                                          "command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                                          "required": ["command"], "additionalProperties": False}}},
]
RIGMARK = ("Return a JSON array of exactly 50 objects describing fictional library books. Each object has the keys "
           "id (integer, 1 to 50 in order), title (string), author (string), year (integer), genres (array of 1 to 3 "
           "strings) and available (boolean). Output only the JSON.")
RIGMARK_SCHEMA = {"type": "array", "minItems": 50, "maxItems": 50, "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "title": {"type": "string"},
                                     "author": {"type": "string"}, "year": {"type": "integer"},
                                     "genres": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                                                "maxItems": 3},
                                     "available": {"type": "boolean"}},
    "required": ["id", "title", "author", "year", "genres", "available"], "additionalProperties": False}}
PLAIN = ["Write a Python function that reverses a linked list.",
         "Explain in four sentences why the sky is blue.",
         "List five prime numbers and their squares."]
LEAK = re.compile(r"</?arg_(key|value)>|</?tool_call>|</?think>")


def post(base: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def validate(value: Any, schema: dict) -> str | None:
    try:
        import jsonschema
    except ImportError:
        return None
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def body_for(a, prompt: str, *, schema: dict | None = None, greedy: bool = True, thinking: bool = False,
             draft: bool = True, max_tokens: int = 1024, extra: dict | None = None) -> dict:
    b: dict = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": thinking}}
    if thinking:
        b["chat_template_kwargs"]["reasoning_effort"] = "low"
    b.update({"temperature": 0} if greedy else {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 20260930})
    if schema is not None:
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    if not draft:
        b["draft"] = False
    b.update(extra or {})
    return b


def sha(r: dict) -> str:
    return str((r.get("tensorfold") or {}).get("sha256"))


def suite_schemas(a) -> tuple[dict, list[str]]:
    fails, rows = [], []
    for greedy in (True, False):
        for thinking in (False, True):
            bodies = [body_for(a, p, schema=s, greedy=greedy, thinking=thinking) for p, s in zip(PROMPTS, SCHEMAS)]
            alone = [post(a.base, b) for b in bodies]
            serial = [post(a.base, dict(b, draft=False)) for b in bodies]
            with cf.ThreadPoolExecutor(4) as ex:
                together = list(ex.map(lambda b: post(a.base, b), bodies))
            for i, (x, y, z) in enumerate(zip(alone, serial, together)):
                tag = f"schema{i} greedy={greedy} thinking={thinking}"
                msg = x["choices"][0]["message"]
                row = {"tag": tag, "sha": sha(x), "finish": x["choices"][0]["finish_reason"],
                       "tokens": x["usage"]["completion_tokens"], "grammar": (x.get("tensorfold") or {}).get("grammar")}
                if not (sha(x) == sha(y) == sha(z)):
                    fails.append(f"{tag}: drafted {sha(x)} serial {sha(y)} batched {sha(z)}")
                if row["finish"] == "stop":
                    try:
                        value = json.loads(msg.get("content") or "")
                    except json.JSONDecodeError as exc:
                        fails.append(f"{tag}: not JSON ({exc.msg})")
                    else:
                        why = validate(value, SCHEMAS[i])
                        if why:
                            fails.append(f"{tag}: schema: {why}")
                if LEAK.search(msg.get("content") or ""):
                    fails.append(f"{tag}: markup in the content")
                rows.append(row)
    return {"rows": rows}, fails


def rate(r: dict) -> float | None:
    tf = r.get("tensorfold") or {}
    return round(r["usage"]["completion_tokens"] / tf["decode_s"], 2) if tf.get("decode_s") else None


def suite_rigmark(a) -> tuple[dict, list[str]]:
    fails, out = [], {}
    for name, schema in (("free", None), ("schema", RIGMARK_SCHEMA)):
        runs = []
        for _ in range(a.reps):
            r = post(a.base, body_for(a, RIGMARK, schema=schema, thinking=True, max_tokens=4096,
                                      extra={"seed": 20260905, "top_p": 1}))
            content = r["choices"][0]["message"].get("content") or ""
            try:
                value = json.loads(content)
                ok = isinstance(value, list) and len(value) == 50 and validate(value, RIGMARK_SCHEMA) is None
            except json.JSONDecodeError:
                ok = False
            if schema is not None and not ok:
                fails.append(f"rigmark {name}: not 50 valid objects")
            runs.append({"valid": ok, "tokens": r["usage"]["completion_tokens"], "tok_s": rate(r), "sha": sha(r),
                         "tokens_per_round": (r.get("tensorfold") or {}).get("tokens_per_round"),
                         "grammar": (r.get("tensorfold") or {}).get("grammar")})
        out[name] = runs
        if len({x["sha"] for x in runs}) != 1:
            fails.append(f"rigmark {name}: replies differ between runs")
    return out, fails


def suite_tools(a) -> tuple[dict, list[str]]:
    fails, rows = [], []
    strict = [dict(t, function=dict(t["function"], strict=True)) for t in TOOLS]
    cases = [("required", TOOLS, "required", "What is the weather in Oslo in Celsius?"),
             ("named", TOOLS, {"type": "function", "function": {"name": "bash"}}, "Show the disk usage of /tmp."),
             ("strict-auto", strict, "auto", "Please check the weather in Kyoto, Fahrenheit.")]
    for name, tools, choice, prompt in cases:
        for thinking in (False, True):
            b = body_for(a, prompt, thinking=thinking, extra={"tools": tools, "tool_choice": choice})
            r = post(a.base, b)
            msg = r["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            row = {"case": name, "thinking": thinking, "calls": len(calls), "finish": r["choices"][0]["finish_reason"]}
            if name != "strict-auto" and not calls:
                fails.append(f"tools {name}: no call")
            for c in calls:
                fn = c["function"]
                known = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
                if fn["name"] not in known:
                    fails.append(f"tools {name}: unknown tool {fn['name']}")
                    continue
                args = json.loads(fn["arguments"])
                why = validate(args, known[fn["name"]])
                if why:
                    fails.append(f"tools {name}: {why}")
                if any(isinstance(v, str) and LEAK.search(v) for v in args.values()):
                    fails.append(f"tools {name}: markup leaked into an argument")
            rows.append(row)
    return {"rows": rows}, fails


def suite_plain(a) -> tuple[dict, list[str]]:
    fails, got = [], {}
    for i, p in enumerate(PLAIN):
        for greedy in (True, False):
            r = post(a.base, body_for(a, p, greedy=greedy, thinking=True, max_tokens=512))
            got[f"{i}-{'g' if greedy else 's'}"] = sha(r)
            if (r.get("tensorfold") or {}).get("grammar") is not None:
                fails.append(f"plain {i}: a grammar ran for an unconstrained request")
    if a.write_ref:
        with open(a.write_ref, "w") as f:
            json.dump(got, f, indent=1)
    if a.ref:
        with open(a.ref) as f:
            ref = json.load(f)
        for k, v in got.items():
            if ref.get(k) != v:
                fails.append(f"plain {k}: sha {v} != knob-off {ref.get(k)}")
    return got, fails


SUITES = {"schemas": suite_schemas, "rigmark": suite_rigmark, "tools": suite_tools, "plain": suite_plain}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--write-ref", default="")
    a = ap.parse_args(argv)
    rec: dict = {"t0": time.time(), "suites": {}, "fails": []}
    for name in a.suites.split(","):
        t0 = time.time()
        res, fails = SUITES[name](a)
        rec["suites"][name] = {"result": res, "fails": fails, "s": round(time.time() - t0, 1)}
        rec["fails"] += fails
        print(f"[structured] {name}: {'PASS' if not fails else 'FAIL'} ({len(fails)} failures, "
              f"{rec['suites'][name]['s']} s)", file=sys.stderr)
        for f_ in fails[:20]:
            print(f"  - {f_}", file=sys.stderr)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    return 0 if not rec["fails"] else 1


if __name__ == "__main__":
    sys.exit(main())
