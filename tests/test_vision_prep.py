"""patches/0500: image preprocessing, fetching and prompt placement (``vision_prep.py``), host only.

- The processor: ``preprocess`` == transformers' ``Glm5NextImageProcessor`` (the torchvision backend that HF's
  ``AutoProcessor`` and vLLM load) bit for bit on the patches and the grid, over sizes from 1 x 1 to past the
  8,000-token budget and extreme aspect ratios; ``smart_resize`` / ``image_tokens`` == transformers' on a sweep of
  sizes; the token counts == ``Glm5NextProcessor._get_num_multimodal_tokens``. Skipped without transformers.
- Without torchvision, the same bits from ``interpolate``.
- Content parts (every accepted shape), ``data:`` URLs, http(s) downloads from a local server (size cap, errors,
  GLM53_TF_VISION_FETCH=0, schemes refused), decode limits (pixels, garbage), RGBA composited on white (vLLM).
- The prompt: sentinels -> ``<|begin_of_image|><|image|><|end_of_image|>`` in prompt order, rows spliced in, a
  placeholder smuggled in the text refused, counts checked.
- Virtual ids: >= 2^24 and < 2^31, a function of the pixels (PNG and BMP of the same pixels agree; one changed
  pixel changes every row), distinct within an image, the registry re-salting a clash.

Run: PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_vision_prep.py
"""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

vp = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision_prep")

PROC = {"do_rescale": True, "patch_expand_factor": 1, "merge_size": 2, "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711], "temporal_patch_size": 2, "patch_size": 14,
        "min_image_tokens": 16, "max_image_tokens": 8000}
SIZES = [(1, 1), (10, 10), (28, 28), (29, 29), (56, 57), (640, 480), (1024, 768), (768, 1024), (1920, 1080),
         (333, 777), (3000, 50), (50, 3000), (4000, 3000), (2000, 2000), (5000, 20), (12000, 9000)]


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    (d / "processor_config.json").write_text(json.dumps({"image_processor": dict(PROC)}))
    (d / "config.json").write_text(json.dumps({"image_token_id": 154854, "image_start_token_id": 154830,
                                               "image_end_token_id": 154831, "vision_config": {"depth": 24}}))
    return vp.Settings.read(d)


def rand_image(w, h, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    ch = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    a = rng.integers(0, 256, (h, w, ch), dtype=np.uint8)
    return Image.fromarray(a[..., 0] if ch == 1 else a, mode)


def data_url(img, fmt="PNG"):
    b = io.BytesIO()
    img.save(b, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()


# -- the processor -------------------------------------------------------------------------------------------------
def _hf():
    ip = pytest.importorskip("transformers.models.glm5_next.image_processing_glm5_next")
    return ip.Glm5NextImageProcessor(**PROC)


@pytest.mark.parametrize("w,h", SIZES)
def test_patches_equal_transformers(settings, w, h):
    pytest.importorskip("torchvision")
    hf = _hf()
    img = rand_image(w, h, seed=w * 7 + h)
    ref = hf(images=[img], return_tensors="pt")
    ours = vp.preprocess(img, settings)
    assert list(ours.grid) == ref["image_grid_thw"][0].tolist()
    assert torch.equal(ours.pixels, ref["pixel_values"])            # bit for bit
    assert ours.tokens == ref["pixel_values"].shape[0] // 4 == vp.image_tokens(h, w, settings)


def test_token_counts_equal_the_processor(settings):
    proc_mod = pytest.importorskip("transformers.models.glm5_next.processing_glm5_next")
    hf = _hf()
    proc = object.__new__(proc_mod.Glm5NextProcessor)
    proc.image_processor = hf
    sizes = [(h, w) for h in (1, 13, 28, 100, 479, 768, 1080, 2160, 7000) for w in (1, 17, 28, 333, 1024, 4096, 9000)]
    got = proc._get_num_multimodal_tokens(image_sizes=sizes)
    assert got["num_image_tokens"] == [vp.image_tokens(h, w, settings) for h, w in sizes]


def test_smart_resize_equals_transformers():
    ip = pytest.importorskip("transformers.models.glm5_next.image_processing_glm5_next")
    rng = np.random.default_rng(3)
    for _ in range(3000):
        h, w = (int(v) for v in rng.integers(1, 12000, 2))
        lo, hi = int(rng.integers(1, 64)), int(rng.integers(64, 16000))
        assert vp.smart_resize(2, h, w, 2, 28, lo, hi) == ip.smart_resize(2, h, w, 2, 28, lo, hi)


def test_budget_caps_and_detail(settings, monkeypatch):
    s = vp.Settings(**{k: getattr(settings, k) for k in ("mean", "std", "patch", "merge", "temporal", "min_tokens",
                                                          "max_tokens", "image_token", "begin_token", "end_token")})
    assert s.budget(None) == 8000 and s.budget("low") == 8000          # detail ignored by default (as vLLM)
    s.low_tokens, s.cap_tokens = 256, 4000
    assert s.budget("low") == 256 and s.budget("high") == 4000 and s.budget("auto") == 4000
    img = rand_image(4000, 3000)
    assert vp.preprocess(img, s, s.budget("low")).tokens <= 256
    assert vp.preprocess(img, s, s.budget(None)).tokens <= 4000


def test_interpolate_fallback_equals_torchvision(settings, monkeypatch):
    pytest.importorskip("torchvision")
    import builtins

    imgs = [rand_image(w, h, seed=w + h) for w, h in [(1024, 768), (3000, 50), (10, 10), (4000, 3000)]]
    with_tv = [vp.preprocess(i, settings).pixels for i in imgs]
    real = builtins.__import__

    def no_tv(name, *a, **k):
        if name.startswith("torchvision"):
            raise ImportError("no torchvision")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_tv)
    for img, ref in zip(imgs, with_tv):
        assert torch.equal(vp.preprocess(img, settings).pixels, ref)


# -- parts, fetch, decode ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("part,url,detail", [
    ({"type": "image_url", "image_url": {"url": "data:x", "detail": "LOW"}}, "data:x", "low"),
    ({"type": "image_url", "image_url": "https://h/i.png"}, "https://h/i.png", None),
    ({"type": "input_image", "image_url": "data:y", "detail": "high"}, "data:y", "high"),
    ({"type": "input_image", "image_url": {"url": "data:z"}}, "data:z", None),
    ({"type": "image", "image": "data:w"}, "data:w", None),
    ({"type": "image", "url": "data:v"}, "data:v", None),
])
def test_part_shapes(part, url, detail):
    ref = vp.part_ref(part)
    assert (ref.url, ref.detail) == (url, detail)


@pytest.mark.parametrize("part", [{"type": "image_url"}, {"type": "image_url", "image_url": {"url": 3}},
                                  {"type": "image_url", "image_url": {}}, {"type": "image", "image": ""}])
def test_malformed_parts(part):
    with pytest.raises(vp.VisionError):
        vp.part_ref(part)


def test_other_parts_are_not_images():
    for p in ({"type": "text", "text": "x"}, "plain", {"type": "video_url", "video_url": {"url": "x"}}, None):
        assert vp.part_ref(p) is None


def test_data_urls(settings):
    img = rand_image(40, 30)
    raw = vp.load_bytes(data_url(img), settings)
    assert Image.open(io.BytesIO(raw)).size == (40, 30)
    assert vp.load_bytes("data:text/plain,hello%20there", settings) == b"hello there"
    with pytest.raises(vp.VisionError, match="comma"):
        vp.load_bytes("data:image/png;base64", settings)
    with pytest.raises(vp.VisionError, match="decode"):
        vp.decode(vp.load_bytes("data:image/png;base64,aGVsbG8=", settings), settings)
    small = vp.Settings(max_bytes=100)
    with pytest.raises(vp.VisionError, match="MAX_BYTES"):
        vp.load_bytes(data_url(rand_image(64, 64)), small)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://h/x.png", "/tmp/x.png", "gopher://h"])
def test_schemes_refused(settings, url):
    with pytest.raises(vp.VisionError, match="scheme"):
        vp.load_bytes(url, settings)


@pytest.fixture
def image_server():
    png = io.BytesIO()
    rand_image(50, 40).save(png, "PNG")
    blobs = {"/ok.png": png.getvalue(), "/big.png": b"\0" * 5000}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path not in blobs:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")          # patches/0600: a declared image type
            self.send_header("Content-Length", str(len(blobs[self.path])))
            self.end_headers()
            self.wfile.write(blobs[self.path])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_http_fetch(settings, image_server):
    # patches/0600: a local http:// server needs the local-testing knobs (HTTPS / public addresses only by default)
    with pytest.raises(vp.VisionError, match="HTTPS on port 443"):
        vp.load_bytes(image_server + "/ok.png", settings)
    local = vp.Settings(fetch_http=True, fetch_private=True)
    with pytest.raises(vp.VisionError, match="public internet addresses"):
        vp.load_bytes(image_server + "/ok.png", vp.Settings(fetch_http=True))
    img = vp.decode(vp.load_bytes(image_server + "/ok.png", local), local)
    assert img.size == (50, 40) and img.mode == "RGB"
    with pytest.raises(vp.VisionError, match="could not fetch"):
        vp.load_bytes(image_server + "/missing.png", local)
    with pytest.raises(vp.VisionError, match="MAX_BYTES"):
        vp.load_bytes(image_server + "/big.png", vp.Settings(max_bytes=1000, fetch_http=True, fetch_private=True))
    with pytest.raises(vp.VisionError, match="FETCH=0"):
        vp.load_bytes(image_server + "/ok.png", vp.Settings(fetch=False))


def test_env_settings(monkeypatch, tmp_path):
    for k, v in {"GLM53_TF_VISION_MAX_IMAGES": "3", "GLM53_TF_VISION_MAX_TOKENS": "2000", "GLM53_TF_VISION_FETCH": "0",
                 "GLM53_TF_VISION_MAX_BYTES": "1234", "GLM53_TF_VISION_LOW_TOKENS": "64"}.items():
        monkeypatch.setenv(k, v)
    s = vp.Settings.read(tmp_path)
    assert (s.max_images, s.cap_tokens, s.fetch, s.max_bytes, s.low_tokens) == (3, 2000, False, 1234, 64)
    monkeypatch.setenv("GLM53_TF_VISION", "2")
    with pytest.raises(ValueError):
        vp.enabled()
    monkeypatch.delenv("GLM53_TF_VISION")
    assert vp.enabled() is False


def test_decode_limits_and_modes(settings):
    with pytest.raises(vp.VisionError, match="MAX_PIXELS"):
        vp.decode(vp.load_bytes(data_url(rand_image(300, 300)), settings), vp.Settings(max_pixels=10000))
    rgba = Image.new("RGBA", (4, 4), (10, 20, 30, 0))
    rgba.putpixel((1, 1), (200, 100, 50, 255))
    out = vp.decode(vp.load_bytes(data_url(rgba), settings), settings)
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255) and out.getpixel((1, 1)) == (200, 100, 50)
    grey = vp.decode(vp.load_bytes(data_url(rand_image(8, 8, mode="L")), settings), settings)
    assert grey.mode == "RGB"
    gif = io.BytesIO()
    frames = [rand_image(8, 8, seed=i).convert("P") for i in range(3)]
    frames[0].save(gif, "GIF", save_all=True, append_images=frames[1:])
    first = vp.decode(gif.getvalue(), settings)
    assert first.size == (8, 8) and first.mode == "RGB"


# -- the prompt ----------------------------------------------------------------------------------------------------
def test_markup_splice_order(settings):
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:a"}},
                                         {"type": "text", "text": "and"},
                                         {"type": "image_url", "image_url": {"url": "data:b"}}]},
            {"role": "user", "content": [{"type": "input_image", "image_url": "data:c"}]}]
    marked, refs = vp.markup(msgs, "n1")
    assert [r.url for r in refs] == ["data:a", "data:b", "data:c"]
    assert msgs[1]["content"][0]["type"] == "image_url"              # the request is not modified
    # a template that renders the last user message first: the images come back in prompt order
    rendered = "".join(p["text"] for p in marked[2]["content"]) + "|" + "".join(p["text"] for p in marked[1]["content"])
    text, order = vp.splice(rendered, "n1", 3, settings)
    assert order == [2, 0, 1]
    assert text == vp.PLACEHOLDER + "|" + vp.PLACEHOLDER + "and" + vp.PLACEHOLDER
    assert vp.count_images(msgs) == 3 and vp.count_images("x") == 0


# -- patches/0660: the image limit and GLM53_TF_VISION_OVERFLOW ----------------------------------------------------
def _img(u):
    return {"type": "image_url", "image_url": {"url": u}}


def test_overflow_env(monkeypatch, tmp_path):
    assert vp.Settings.read(tmp_path).overflow == "refuse"
    monkeypatch.setenv("GLM53_TF_VISION_OVERFLOW", " Drop_Oldest ")
    assert vp.Settings.read(tmp_path).overflow == "drop_oldest"
    monkeypatch.setenv("GLM53_TF_VISION_OVERFLOW", "drop")
    with pytest.raises(ValueError, match="GLM53_TF_VISION_OVERFLOW"):
        vp.Settings.read(tmp_path)


def test_drop_oldest_keeps_the_newest():
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [_img("data:a"), {"type": "text", "text": "x"}, _img("data:b")]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "input_image", "image_url": "data:c"}, _img("data:d")]}]
    out, n = vp.drop_oldest(msgs, 3)
    assert n == 1 and [r.url for r in vp.markup(out, "z")[1]] == ["data:b", "data:c", "data:d"]
    assert out[1]["content"][0] == {"type": "text", "text": vp.OMITTED} and out[1]["content"][1:] == msgs[1]["content"][1:]
    assert msgs[1]["content"][0]["type"] == "image_url"              # the request is not modified
    assert out[0] is msgs[0] and out[3] is msgs[3]
    out, n = vp.drop_oldest(msgs, 1)
    assert n == 3 and [r.url for r in vp.markup(out, "z")[1]] == ["data:d"] and vp.count_images(out) == 1
    assert vp.drop_oldest(msgs, 4) == (msgs, 0) and vp.drop_oldest(msgs, 9) == (msgs, 0)
    assert vp.last_images(msgs) == 2 and vp.last_images([]) == 0


def test_too_many_says_the_history_counts():
    s = vp.Settings(max_images=8)
    msg = str(vp.too_many(9, s))
    assert "9 images" in msg and "at most 8" in msg and "earlier turns" in msg
    for knob in ("GLM53_TF_VISION_MAX_IMAGES", "GLM53_TF_VISION_OVERFLOW=drop_oldest"):
        assert knob in msg
    assert "10 images in the last message" in str(vp.too_many(12, s, 10))


def _prep(n, digest=b"d"):
    return vp.Prepared(None, (1, 2, 2 * n), n, digest, (1, 1), vp.derive(digest, n))


def test_expand_rows_and_refusals(settings):
    B, I, E = settings.begin_token, settings.image_token, settings.end_token
    a, b = _prep(3, b"a"), _prep(2, b"b")
    ids = [1, B, I, E, 2, B, I, E, 3]
    assert vp.expand(ids, [a, b], settings) == [1, B, *a.vids, E, 2, B, *b.vids, E, 3]
    assert vp.expand(ids, [a, b], settings, virtual=False) == [1, B, I, I, I, E, 2, B, I, I, E, 3]
    with pytest.raises(vp.VisionError, match="not an image part"):
        vp.expand([1, I, 2, B, I, E], [a, b], settings)                   # a bare <|image|> typed in the text
    with pytest.raises(vp.VisionError, match="not an image part"):
        vp.expand(ids + [B, I, E], [a, b], settings)                      # one more framed placeholder than images
    with pytest.raises(vp.VisionError, match="placed"):
        vp.expand(ids, [a, b, a], settings)


def test_vids_are_a_function_of_the_pixels(settings):
    img = rand_image(120, 90, seed=5)
    p_png = vp.preprocess(vp.decode(vp.load_bytes(data_url(img, "PNG"), settings), settings), settings)
    p_bmp = vp.preprocess(vp.decode(vp.load_bytes(data_url(img, "BMP"), settings), settings), settings)
    assert p_png.digest == p_bmp.digest
    other = img.copy()
    other.putpixel((0, 0), tuple((v + 1) % 256 for v in img.getpixel((0, 0))))
    p_other = vp.preprocess(other, settings)
    assert p_other.digest != p_png.digest
    v1, v2 = vp.derive(p_png.digest, p_png.tokens), vp.derive(p_other.digest, p_other.tokens)
    assert len(set(v1)) == len(v1) and all(vp.VBASE <= v <= vp.VLIMIT for v in v1 + v2)
    assert all(x != y for x, y in zip(v1, v2))                            # every row differs, not just the first
    # the digest is the canvas's: a budget the image fits in changes nothing, one that shrinks it changes every row
    assert vp.preprocess(img, settings, 64).digest == p_png.digest and p_png.tokens == 20
    assert vp.preprocess(img, settings, 16).digest != p_png.digest


def test_registry_resalts_a_clash(monkeypatch):
    reg = vp.Registry()
    real = vp.derive
    calls = []

    def fake(digest, n, salt=0):
        calls.append((digest, salt))
        if salt == 0:
            return [vp.VBASE + i for i in range(n)]                     # every image's first derivation collides
        return real(digest, n, salt)

    monkeypatch.setattr(vp, "derive", fake)
    a = reg.vids(b"A", 4)
    b = reg.vids(b"B", 4)
    assert a == [vp.VBASE + i for i in range(4)] and set(a).isdisjoint(b) and (b"B", 1) in calls
    assert reg.vids(b"B", 4) == b and reg.vids(b"A", 4) == a            # stable while remembered
