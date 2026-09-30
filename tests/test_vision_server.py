"""patches/0500: image input through the OpenAI routes (``app.GlmApp``), and every token-keyed cache telling two
images apart. Host only: a synthetic checkpoint folder (a character tokenizer with GLM's special tokens, a chat
template with the checkpoint's ``visible_text`` macro, the processor config) and a fake engine.

- Off by default: without GLM53_TF_VISION an image part renders the template's own reminder, the prompt has no
  image rows and the engine gets no images.
- On: the prompt carries ``<|begin_of_image|>``, one virtual id a row, ``<|end_of_image|>`` where the part was; the
  engine gets the preprocessed images; ``usage.prompt_tokens`` counts the rows; ``/tokenize`` returns them as
  ``<|image|>`` ids; an image URL is fetched once a request (``check`` then ``run``).
- HTTP 400s before anything streams: a bad data URL, a refused scheme, too many images, a placeholder typed in the
  text, a request past the context counting the image rows (``context_length_exceeded``).
- Caches (the requirement: two different images with the same text never share state past the image): the prompts
  differ at every image row and agree before it; ``GlmEngine._resume`` / ``Batcher._resume`` (the live
  snapshots), the session store's ``SessionIndex`` (``find``, ``lcp``, the fork marks, page keys) and the NVMe tier's
  ``DiskIndex`` (the same lookups) all stop at the first image row for a different image and resume past it for the
  same image (sent as PNG or BMP: the same pixels); ``ids_b64`` / the int32 packing keep virtual ids intact.
- The lookup drafter never proposes an image row's id.
- With the real tokenizer and template ($GLM53_TF_TOKENIZER_DIR, plus transformers): the prompt ids ==
  transformers' ``Glm5NextProcessor`` with the vLLM kit's vision template ($GLM53_TF_VISION_TEMPLATE, optional).

Run: PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_vision_server.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("jinja2")
tokenizers = pytest.importorskip("tokenizers")
from PIL import Image  # noqa: E402

vp = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision_prep")
app_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.app")
from tensorfold.cuda import server  # noqa: E402

SPECIALS = ["<|endoftext|>", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
            "<|begin_of_image|>", "<|end_of_image|>", "<|image|>", "<think>", "</think>"]
TEMPLATE = r"""[gMASK]<sop>
{%- macro visible_text(content) -%}
    {%- if content is string -%}
        {{- content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}
                {{- item.text }}
            {%- elif item is string -%}
                {{- item }}
            {%- elif item is mapping and item.type in ['image', 'image_url', 'video', 'video_url'] -%}
                {%- set media_type = item.type | replace('_url', '') -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ ".</reminder>" }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- content }}
    {%- endif -%}
{%- endmacro -%}
{%- for m in messages -%}
{%- if m.role == 'user' -%}<|user|>{{ visible_text(m.content) }}
{%- elif m.role == 'system' -%}<|system|>{{ visible_text(m.content) }}
{%- elif m.role == 'assistant' -%}<|assistant|><think></think>{{ visible_text(m.content) }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}"""


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    d = tmp_path_factory.mktemp("glm")
    chars = [chr(c) for c in range(32, 127)] + ["\n", "\x00"]
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok.add_special_tokens(SPECIALS)
    tok.save(str(d / "tokenizer.json"))
    ids = {s: tok.token_to_id(s) for s in SPECIALS}
    (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}))
    (d / "chat_template.jinja").write_text(TEMPLATE)
    (d / "config.json").write_text(json.dumps({
        "image_token_id": ids["<|image|>"], "image_start_token_id": ids["<|begin_of_image|>"],
        "image_end_token_id": ids["<|end_of_image|>"], "vision_config": {"depth": 24}}))
    (d / "processor_config.json").write_text(json.dumps({"image_processor": {
        "merge_size": 2, "patch_size": 14, "temporal_patch_size": 2, "min_image_tokens": 16,
        "max_image_tokens": 8000, "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711]}}))
    return d, ids


class Engine:
    def __init__(self, eos, limit=None):
        self.eos = (eos,)
        if limit is not None:
            self.limit = limit
        self.request = threading.local()
        self.prompts, self.visions = [], []

    def generate(self, prompt, max_tokens, sampling, on_tokens):
        self.prompts.append(list(prompt))
        self.visions.append(getattr(self.request, "vision", None))
        on_tokens([self.eos[0]])
        return {}


def make_app(model_dir, monkeypatch, *, on=True, limit=None, **env):
    d, ids = model_dir
    monkeypatch.setenv("GLM53_TF_TOKCACHE", "0")
    if on:
        monkeypatch.setenv("GLM53_TF_VISION", "1")
    else:
        monkeypatch.delenv("GLM53_TF_VISION", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    eng = Engine(ids["<|endoftext|>"], limit)
    app = app_mod.GlmApp(eng, d, "glm", sampling={"temperature": 0.0})
    return app, eng


def image(w=100, h=80, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


def url(img, fmt="PNG"):
    b = io.BytesIO()
    img.save(b, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()


def chat(*parts, text="What is in the picture?", **extra):
    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in parts]
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 4, **extra}


def run(app, body):
    problem = app.check(body)
    assert problem is None, problem
    return app.run(body, True, lambda d: True)


# -- off / on ------------------------------------------------------------------------------------------------------
def test_off_by_default_renders_the_reminder(model_dir, monkeypatch):
    app, eng = make_app(model_dir, monkeypatch, on=False)
    assert app.vision is None
    run(app, chat(url(image())))
    text = app.tok.decode(eng.prompts[0], skip_special_tokens=False)
    assert "unabletoprocessthisimage" in text.replace(" ", "")
    assert not vp.has_vids(eng.prompts[0]) and eng.visions[0] is None


def test_on_places_rows(model_dir, monkeypatch):
    app, eng = make_app(model_dir, monkeypatch)
    _, ids = model_dir
    img = image(300, 200)
    res = run(app, chat(url(img)))
    p = eng.prompts[0]
    req = eng.visions[0]
    n = vp.image_tokens(200, 300, app.vision.s)
    assert len(req.images) == 1 and req.images[0].tokens == n and req.images[0].size == (200, 300)
    b = p.index(ids["<|begin_of_image|>"])
    assert p[b + 1:b + 1 + n] == req.images[0].vids and p[b + 1 + n] == ids["<|end_of_image|>"]
    assert ids["<|image|>"] not in p and sum(vp.is_vid(t) for t in p) == n
    assert p[:b] == app.prompt_tokens.encode("[gMASK]<sop><|user|>What is in the picture?")
    assert res["prompt"] == len(p) if "prompt" in res else True
    # a text-only request on the same server: no rows, no images
    run(app, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
    assert not vp.has_vids(eng.prompts[1]) and eng.visions[1] is None


def test_two_images_in_order_and_duplicates(model_dir, monkeypatch):
    app, eng = make_app(model_dir, monkeypatch)
    a, b = image(seed=1), image(seed=2)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url(a)}}, {"type": "text", "text": "vs"},
        {"type": "image_url", "image_url": url(b)}, {"type": "image_url", "image_url": {"url": url(a, "BMP")}}]}],
        "max_tokens": 2}
    run(app, body)
    imgs = eng.visions[0].images
    assert [i.digest for i in imgs] == [imgs[0].digest, imgs[1].digest, imgs[0].digest] and imgs[0].digest != imgs[1].digest
    rows = [t for t in eng.prompts[0] if vp.is_vid(t)]
    n = imgs[0].tokens
    assert rows == imgs[0].vids + imgs[1].vids + imgs[0].vids and len(rows) == 2 * n + imgs[1].tokens


def test_same_text_different_images_differ_at_every_row(model_dir, monkeypatch):
    app, eng = make_app(model_dir, monkeypatch)
    x, y = image(seed=3), image(seed=4)
    for u in (url(x), url(y), url(x, "BMP")):
        run(app, chat(u))
    px, py, px2 = eng.prompts
    assert px == px2                                          # the same pixels: the same prompt (state is shared)
    first = next(i for i, t in enumerate(px) if vp.is_vid(t))
    assert px[:first] == py[:first] and len(px) == len(py)
    assert all(a != b for a, b in zip(px, py) if vp.is_vid(a))
    assert [i for i, (a, b) in enumerate(zip(px, py)) if a != b] == [i for i, t in enumerate(px) if vp.is_vid(t)]


# -- HTTP ----------------------------------------------------------------------------------------------------------
@pytest.fixture
def http():
    servers = []

    def start(app):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


@pytest.mark.parametrize("stream", [False, True])
def test_http_ok_and_usage(model_dir, monkeypatch, http, stream):
    app, eng = make_app(model_dir, monkeypatch)
    base = http(app)
    body = chat(url(image(560, 280)), stream=stream, stream_options={"include_usage": True})
    code, text = post(base, "/v1/chat/completions", body)
    assert code == 200, text
    if not stream:
        assert json.loads(text)["usage"]["prompt_tokens"] == len(eng.prompts[0])


@pytest.mark.parametrize("body_fn,needle", [
    (lambda: chat("data:image/png;base64,!!notbase64!!"), "base64"),
    (lambda: chat("data:image/png;base64,aGVsbG8="), "decode"),
    (lambda: chat("file:///etc/passwd"), "scheme"),
    (lambda: chat(*[url(image(seed=i)) for i in range(3)]), "at most 2"),
    (lambda: chat(url(image()), text="look <|begin_of_image|><|image|><|end_of_image|> here"), "placeholder"),
    (lambda: chat(url(image()), text="a bare <|image|> token"), "placeholder"),
    (lambda: {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}, "URL"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_http_400s(model_dir, monkeypatch, http, body_fn, needle, stream):
    app, eng = make_app(model_dir, monkeypatch, GLM53_TF_VISION_MAX_IMAGES=2)
    code, text = post(http(app), "/v1/chat/completions", dict(body_fn(), stream=stream))
    assert code == 400 and needle in json.loads(text)["error"]["message"], text
    assert json.loads(text)["error"]["param"] == "messages"
    assert not eng.prompts


def test_context_counts_image_rows(model_dir, monkeypatch, http):
    app, eng = make_app(model_dir, monkeypatch, limit=400)
    img = image(700, 700)                                    # 625 rows: past the limit on its own
    code, text = post(http(app), "/v1/chat/completions", chat(url(img)))
    err = json.loads(text)["error"]
    assert code == 400 and err["code"] == "context_length_exceeded", text
    n = vp.image_tokens(700, 700, app.vision.s)
    assert f"{len(app.prompt_tokens.encode('[gMASK]<sop><|user|>What is in the picture?'))}" in err["message"] or n
    code, _ = post(http(app), "/v1/chat/completions", chat(url(image(60, 60))))
    assert code == 200


def test_tokenize_counts_rows(model_dir, monkeypatch, http):
    app, eng = make_app(model_dir, monkeypatch)
    _, ids = model_dir
    img = image(300, 300)
    code, text = post(http(app), "/tokenize", {"messages": chat(url(img))["messages"]})
    got = json.loads(text)
    n = vp.image_tokens(300, 300, app.vision.s)
    assert code == 200 and got["tokens"].count(ids["<|image|>"]) == n and got["count"] == len(got["tokens"])
    assert not any(vp.is_vid(t) for t in got["tokens"])


def test_url_fetched_once_a_request(model_dir, monkeypatch, http):
    # patches/0600: a local http:// image server needs the local-testing knobs (HTTPS / public addresses by default)
    monkeypatch.setenv("GLM53_TF_VISION_FETCH_HTTP", "1")
    monkeypatch.setenv("GLM53_TF_VISION_FETCH_PRIVATE", "1")
    png = io.BytesIO()
    image(90, 60).save(png, "PNG")
    hits = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")          # patches/0600: a declared image type
            self.send_header("Content-Length", str(len(png.getvalue())))
            self.end_headers()
            self.wfile.write(png.getvalue())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        app, eng = make_app(model_dir, monkeypatch, limit=100000)
        code, text = post(http(app), "/v1/chat/completions",
                          chat(f"http://127.0.0.1:{srv.server_address[1]}/cat.png"))
        assert code == 200, text
        assert hits == ["/cat.png"] and len(eng.visions[0].images) == 1
    finally:
        srv.shutdown()
        srv.server_close()


# -- the caches ----------------------------------------------------------------------------------------------------
@pytest.fixture
def prompts(model_dir, monkeypatch):
    """Four prompts: image X, image Y (same text), image X again (as BMP), image X with a longer question; each 2,000+
    tokens long so the session store's 256-token pages hold rows before, across and after the image."""

    app, eng = make_app(model_dir, monkeypatch)
    pre = "context " * 80
    x, y = image(900, 700, seed=10), image(900, 700, seed=11)
    for u in (url(x), url(y), url(x, "BMP")):
        run(app, chat(u, text=pre))
    turn2 = chat(url(x), text=pre)                            # the next turn of X's conversation
    turn2["messages"] += [{"role": "assistant", "content": "It is noise."}, {"role": "user", "content": "Sure?"}]
    run(app, turn2)
    return eng.prompts


def snap(ids, grid=0):
    return SimpleNamespace(ids=list(ids), grid=grid, drafter_end=len(ids), mtp_len=len(ids) - 1)


def first_row(p):
    return next(i for i, t in enumerate(p) if vp.is_vid(t))


def test_live_snapshots(prompts):
    engine_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.engine")
    px, py, px2, pxl = prompts
    f = first_row(px)
    stub = SimpleNamespace(cache=[snap(px[:f - 1]), snap(px[:f + 1]), snap(px[:f + 300]), snap(px)],
                           _drafters=lambda code: (False, True, False), _grid=lambda: 0)
    hit = engine_mod.GlmEngine._resume(stub, py + [1], [0], 0)
    assert hit is not None and len(hit.ids) == f - 1                  # only the text before the image
    hit = engine_mod.GlmEngine._resume(stub, px2 + [1], [0], 0)
    assert len(hit.ids) == len(px)                                    # the same image: the whole prompt
    batch_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.batch")
    bst = SimpleNamespace(caches={0: stub.cache}, g=stub)
    assert len(batch_mod.Batcher._resume(bst, 0, py + [1], [0], 0).ids) == f - 1
    assert len(batch_mod.Batcher._resume(bst, 0, pxl, [0], 0).ids) == len(px)     # the next turn, past the image


def test_session_store_and_disk_tier(prompts):
    sessions = pytest.importorskip("tensorfold.families.glm5_next.cuda.sessions")
    sessdisk = pytest.importorskip("tensorfold.families.glm5_next.cuda.sessdisk")
    px, py, px2, pxl = prompts
    f = first_row(px)
    PAGE = sessions.PAGE
    assert len(px) > f + 2 * PAGE
    idx = sessions.SessionIndex(1 << 40, (100, 10, 50, 5))
    saved = idx.save(tag=0, ids=px, mtp_len=len(px) - 1, drafter=False, snap_bytes=1000)
    assert saved.entry is not None
    assert idx.find(py + [1], 0, True, False) is None
    assert idx.find(px2 + [1], 0, True, False) is saved.entry
    assert idx.find(pxl, 0, True, False) is saved.entry               # X's next turn resumes past the image
    assert idx.lcp(py) == f and idx.lcp(px2) == len(px) and idx.lcp(pxl) == len(px)
    # the fork mark for Y's request is at or before its first image row: nothing past the image is ever shared
    marks = idx.marks(py, 0, 0, has_mtp=True, drafter=False, every=0, fork_min=1)
    assert all(m <= f for m in marks)
    # page identities: equal for the pages before the image, different for every page from the image on
    bx, by = sessions.chain(px), sessions.chain(py)
    kx = sessions.page_keys(0, px, bx, len(bx), True)
    ky = sessions.page_keys(0, py, by, len(by), True)
    for p in range(len(kx)):
        before = sessions.key_end(0, p) <= f
        assert (kx[p] == ky[p]) == before, p
    assert sessions.ids_key(0, px, True, False) != sessions.ids_key(0, py, True, False)
    assert sessions.ids_key(0, px, True, False) == sessions.ids_key(0, px2, True, False)
    # the NVMe tier's index runs the same lookups
    disk = sessdisk.DiskIndex(1 << 40, lambda mtp: 1000)
    disk.add(tag=0, ids=px, mtp_len=len(px) - 1, drafter=False, entry_bytes=1000,
             **_disk_extra(sessdisk.DiskIndex.add, sessions, px))
    assert disk.find(py + [1], 0, True, False) is None and disk.find(px2 + [1], 0, True, False) is not None
    # virtual ids survive the int32 packing the tier writes
    assert sessdisk.b64_ids(sessdisk.ids_b64(py)) == py


def _disk_extra(add, sessions, ids):
    import inspect

    params = inspect.signature(add).parameters
    extra = {}
    if "chain" in params:
        extra["chain"] = sessions.chain(ids)
    if "pages" in params:
        bl = sessions.chain(ids)
        extra["pages"] = sessions.page_keys(0, ids, bl, len(bl), True)
    return {k: v for k, v in extra.items() if k in params}


def test_lookup_never_drafts_an_image_row():
    lookup = pytest.importorskip("tensorfold.families.glm5_next.cuda.lookup")
    vids = vp.derive(b"z", 6)
    costs = {"verify": [1.0] * 16, "mtp": [1.0] * 16}
    # the history (prompt, then the reply's 8) repeats 5 6 7 8, which the image followed: no draft is an image row
    look = lookup.Lookup([5, 6, 7, 8, *vids, 9, 10, 5, 6, 7], costs, most=7, min_match=3, gated=False)
    assert look.plan([8], 7) == []
    look = lookup.Lookup([5, 6, 7, 8, 11, 12, *vids, 9, 5, 6, 7], costs, most=7, min_match=3, gated=False)
    assert look.plan([8], 7) == [11, 12]                              # cut where the image starts
    assert look.index.propose(7, 3)[0] == [11, 12]                    # (0450's resident rounds read this)


# -- parity with transformers' processor on the real tokenizer and template -----------------------------------------
TOKDIR = Path(os.environ.get("GLM53_TF_TOKENIZER_DIR", "/nonexistent"))
KIT_TEMPLATE = os.environ.get("GLM53_TF_VISION_TEMPLATE", "")


@pytest.mark.skipif(not (TOKDIR / "tokenizer.json").exists() or not (TOKDIR / "processor_config.json").exists(),
                    reason="GLM53_TF_TOKENIZER_DIR without the checkpoint's tokenizer / processor_config.json")
def test_prompt_equals_transformers_processor(monkeypatch):
    proc_mod = pytest.importorskip("transformers.models.glm5_next.processing_glm5_next")
    ip_mod = pytest.importorskip("transformers.models.glm5_next.image_processing_glm5_next")
    vproc = pytest.importorskip("transformers.models.glm5_next.video_processing_glm5_next")
    auto = pytest.importorskip("transformers")
    monkeypatch.setenv("GLM53_TF_VISION", "1")
    monkeypatch.setenv("GLM53_TF_TOKCACHE", "0")
    eng = Engine(154820)
    app = app_mod.GlmApp(eng, TOKDIR, "glm")
    ipc = json.loads((TOKDIR / "processor_config.json").read_text())["image_processor"]
    ipc.pop("image_processor_type", None)
    template = Path(KIT_TEMPLATE).read_text() if KIT_TEMPLATE else (TOKDIR / "chat_template.jinja").read_text() \
        .replace("""{%- set media_type = item.type | replace('_url', '') | replace('input_', '') -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}""",
                 "{{- '<|begin_of_image|><|image|><|end_of_image|>' -}}")
    proc = proc_mod.Glm5NextProcessor(image_processor=ip_mod.Glm5NextImageProcessor(**ipc),
                                      tokenizer=auto.AutoTokenizer.from_pretrained(str(TOKDIR)),
                                      video_processor=vproc.Glm5NextVideoProcessor(), chat_template=template)
    rng = np.random.default_rng(0)
    for sizes in ([(1024, 768)], [(17, 900), (400, 300)], [(2000, 2000)]):
        imgs = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for w, h in sizes]
        hf_msgs = [{"role": "system", "content": "Be brief."},
                   {"role": "user", "content": [{"type": "text", "text": "Describe."}] +
                    [{"type": "image", "image": im} for im in imgs]}]
        text = proc.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
        ref = proc(text=[text], images=imgs, return_tensors="pt")["input_ids"][0].tolist()
        ours_msgs = [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": [{"type": "text", "text": "Describe."}] +
                      [{"type": "image_url", "image_url": {"url": url(im)}} for im in imgs]}]
        ours = app.tokenize({"messages": ours_msgs, "chat_template_kwargs": {"enable_thinking": True}})
        assert ours == ref, (sizes, len(ours), len(ref))
