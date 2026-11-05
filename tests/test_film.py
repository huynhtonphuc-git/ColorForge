"""Film-stage tests: datasheet round trip, LUT accuracy, GPU == CPU. Run: env\\cf\\python.exe -m pytest tests -q"""
import sys
from pathlib import Path

import colour
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src" / "colorforge"))
import film  # noqa: E402
from densitometry import status_a  # noqa: E402


@pytest.fixture(scope="module")
def k64():
    st = film.Stock(ROOT / "data" / "stocks" / "kodachrome_64.cfstock.json")
    view = colour.sd_blackbody(3200, film.SHAPE).values
    return st, view, film.bake_lut(st, view)


def test_neutral_step_wedge_matches_datasheet(k64):
    """Equal log H in all layers (the datasheet's neutral exposure) must reproduce Kodak's Status A curves."""
    st, view, _ = k64
    m = (st.loge > -2.4) & (st.loge < 0.4)  # drawn range
    logH = np.repeat(st.loge[m, None], 3, 1)
    _, sd = film.render_exact(logH, st, view)
    err = np.abs(status_a(sd) - st.D[m])
    assert err.mean() <= 0.05 and err.max() <= 0.05, f"mean {err.mean():.4f} max {err.max():.4f}"


def test_lut_matches_exact(k64):
    st, view, lut = k64
    rng = np.random.default_rng(0); logH = rng.uniform(-2.6, 0.6, (4000, 3))
    exact, _ = film.render_exact(logH, st, view)
    approx = film.apply_lut_cpu(logH.astype(np.float32), lut)
    lab = lambda rgb: colour.XYZ_to_Lab(np.clip(rgb, 0, None) @ np.linalg.inv(film.XYZ_TO_BT709).T, colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    de = colour.delta_E(lab(exact), lab(approx), method="CIE 2000")
    assert np.percentile(de, 99) < 0.5, f"p99 dE00 {np.percentile(de, 99):.3f}"


def test_acutance_reproduces_datasheet_mtf():
    """Sine targets of log H at datasheet frequencies (sized in um on a 36 mm frame) -> response within +-0.05."""
    if not (ROOT / "data" / "kodak_e55").exists():
        pytest.skip("digitised sheet curves are not in the public copy")
    import acutance
    p = acutance.KODACHROME_64; W = 4096; pitch = acutance.um_per_px((64, W))
    d = np.loadtxt(next((ROOT / "data" / "kodak_e55").glob("e88-2009_06_p4_MTF_*.csv")), delimiter=",", skiprows=1)
    xs = np.arange(W) * pitch / 1000.0  # mm
    for v, r in d[:: max(1, len(d) // 12)]:
        if v > 0.8 * 500 / pitch:  # below 80 % of Nyquist
            continue
        k = round(v * xs[-1]); v_exact = k / (W * pitch / 1000.0)  # whole cycles across the frame: no wrap error
        logH = np.zeros((64, W, 3), np.float32) + (-0.7 + 0.01 * np.sin(2 * np.pi * v_exact * xs))[None, :, None]
        out = acutance.apply(logH, p)
        resp = (out[32, W // 4:3 * W // 4, 1].max() - out[32, W // 4:3 * W // 4, 1].min()) / 0.02
        assert abs(resp - np.interp(v_exact, d[:, 0], d[:, 1] / 100)) <= 0.05, f"{v_exact:.1f} c/mm: {resp:.3f}"


def test_acutance2_small_signal_matches_kodak_mtf(k64):
    """Nonlinear v2 chain, small sine of log H at the reference density, green layer: response within +-0.05 of Kodak."""
    if not (ROOT / "data" / "kodak_e55").exists():
        pytest.skip("digitised sheet curves are not in the public copy")
    import acutance2
    st, _, _ = k64; p = dict(acutance2.KODACHROME_64, halation=(0, 0, 0)); W = 4096; pitch = 36000 / W
    d = np.loadtxt(next((ROOT / "data" / "kodak_e55").glob("e88-2009_06_p4_MTF_*.csv")), delimiter=",", skiprows=1)
    lh0 = float(np.interp(p["ref_density"], st.D[::-1, 1], st.loge[::-1])); xs = np.arange(W) * pitch / 1000.0
    for v, r in d[:: max(1, len(d) // 10)]:
        if v > 0.8 * 500 / pitch:
            continue
        k = round(v * xs[-1]); ve = k / (W * pitch / 1000.0)
        logH = np.zeros((48, W, 3), np.float32) + (lh0 + 0.004 * np.sin(2 * np.pi * ve * xs))[None, :, None]
        out, _ = acutance2.apply(st, logH, p, pitch)
        row = out[24, W // 4:3 * W // 4, 1]; resp = (row.max() - row.min()) / 0.008
        assert abs(resp - np.interp(ve, d[:, 0], d[:, 1] / 100)) <= 0.05, f"{ve:.1f} c/mm: {resp:.3f}"


def test_interimage_keeps_greys_and_matches_fuji_ranges(k64):
    import interimage
    st, view, _ = k64; prm = interimage.KODACHROME_64
    # uniform grey patches: no shift at all
    for lh in np.linspace(-2.2, 0.2, 7):  # uniform grey patches (a ramp is not uniform: the 20 um spread acts on it)
        grey = np.full((32, 32, 3), lh, np.float32)
        out, _ = interimage.apply(st, grey, prm, um_per_px=8.8)
        assert np.abs(out - grey).max() < 1e-5, lh
    # single-colour shifts inside the Fuji EP0442323 ranges (red, green) at D 0.5 and D 1.5
    s05, s15 = interimage.single_colour_shift(st, prm, 0.5), interimage.single_colour_shift(st, prm, 1.5)
    assert 0.20 <= s05[0] <= 0.40 and 0.25 <= s05[1] <= 0.45, s05
    assert s15[0] <= 0.07 + 1e-3 and s15[1] <= 0.15, s15


def test_manual_white_sliders():
    g = film.manual_white(0, 0); assert np.allclose(g, 1)
    w = film.manual_white(100, 0); assert w[0] > 1 > w[2] and abs(np.log2(w[0] / w[2]) - 1) < 1e-5  # warmer: R up, B down, 1 stop apart
    m = film.manual_white(0, 100); assert m[1] < 1 < m[0] and abs(m[0] - m[2]) < 1e-6  # magenta: G down
    assert abs(w @ [0.2627, 0.6780, 0.0593] - 1) < 1e-5  # luminance kept


def test_tone_zones_identity_and_monotonic():
    import studio
    x = np.geomspace(1e-4, 1.2, 2000, dtype=np.float32)[:, None].repeat(3, 1)
    assert studio.tone_zones(x) is x
    Y0 = x @ studio.P3_Y
    for a in (-100, 100):
        for z in ([a, 0, 0], [0, a, 0], [0, 0, a], [a, a, a]):
            y = studio.tone_zones(x, *z); assert (np.diff(y @ studio.P3_Y) > 0).all(), z  # no tone reversal
            assert np.allclose(y[:, 0] / y[:, 1], x[:, 0] / x[:, 1], atol=1e-4)  # chroma ratios kept
    assert abs(float((studio.tone_zones(x, 0, 100, 0) @ studio.P3_Y)[np.argmin(np.abs(np.log2(Y0 / 0.18) + 0.5))]) - float(Y0[np.argmin(np.abs(np.log2(Y0 / 0.18) + 0.5))])) < 2e-3  # mid zone pivots on its centre


def test_contrast_saturation_vibrance():
    import studio
    rng = np.random.default_rng(0); x = rng.uniform(0.01, 0.9, (500, 3)).astype(np.float32)
    assert studio.global_contrast(x) is x and studio.saturate(x) is x
    assert np.allclose(studio.global_contrast(np.full((1, 3), 0.18, np.float32), 100), 0.18, atol=1e-4)  # pivots on mid grey
    grey = np.geomspace(0.01, 1, 200, dtype=np.float32)[:, None].repeat(3, 1)
    for a in (-100, 100):
        assert (np.diff(studio.global_contrast(grey, a) @ studio.P3_Y) > 0).all()
        assert np.allclose(studio.saturate(grey, a, a), grey, atol=1e-5)  # greys stay grey
        assert np.allclose(studio.saturate(x, a, 0) @ studio.P3_Y, x @ studio.P3_Y, atol=1e-4)  # luminance kept
    assert np.allclose(studio.saturate(x, -100), (x @ studio.P3_Y)[:, None], atol=1e-5)  # -100 = grey
    muted, vivid = np.array([[0.5, 0.45, 0.4]], np.float32), np.array([[0.9, 0.1, 0.05]], np.float32)
    gain = lambda c: float(np.ptp(studio.saturate(c, 0, 100)) / np.ptp(c))
    assert gain(muted) > 1.5 > 1.05 > gain(vivid) >= 1.0  # vibrance favours muted colours


def test_dng_and_jxl_roundtrip(tmp_path):
    import imagecodecs as ic, ingest, output, rawpy
    rng = np.random.default_rng(0); p3 = rng.uniform(0.02, 0.9, (64, 96, 3)).astype(np.float32)
    d = tmp_path / "t.dng"; output.write_linear_dng(p3, d)
    img, _ = ingest._linear_dng(d)
    want = p3 @ (film.XYZ_TO_BT2020 @ output.rgb_to_xyz(output.SPACES["p3"])).T
    assert np.abs(img - want).max() < 0.004, np.abs(img - want).max()
    rawpy.imread(str(d)).close()  # LibRaw recognises the file
    j = tmp_path / "t.jxl"; output.write_jxl(p3, j); back = ic.jpegxl_decode(j.read_bytes())
    assert back.dtype == np.uint16 and np.abs(back / 65535.0 - output.srgb_oetf(p3)).max() < 2e-5  # lossless


def test_ai_micro_contrast_obeys_the_ai_rule():
    """Only high-frequency brightness may come from the model: broad areas unchanged, field is a single brightness channel, flat areas stay flat."""
    import cv2, enhance
    yy, xx = np.mgrid[0:300, 0:420]; img = np.stack([0.05 + xx / 420 * 0.5, 0.08 + yy / 300 * 0.3, 0.2 * np.ones_like(xx, float)], -1).astype(np.float32)
    img[100:200, 150:280] *= 2.5; img[230:280, 30:90] = np.float32([0.5, 0.1, 0.05])
    m = enhance.MicroContrast(); assert "openvino" in m.backend or "torch" in m.backend
    d = m(img, 8.82); assert d.shape == img.shape[:2] and np.abs(d).max() <= enhance.CLIP_LOG + 1e-6
    assert np.abs(cv2.GaussianBlur(d, (0, 0), 16)).max() < 0.01, np.abs(cv2.GaussianBlur(d, (0, 0), 16)).max()  # low frequencies stay with the physics
    assert np.abs(d[20:80, 300:400]).max() < 0.01, np.abs(d[20:80, 300:400]).max()  # a smooth gradient gets no invented texture


def test_ai_micro_contrast_only_raises_contrast():
    """It must add to the photo's local contrast, never smooth it: the detail field goes the same way as the input's own fine detail."""
    import cv2, enhance
    rng = np.random.default_rng(1); n = cv2.GaussianBlur(rng.standard_normal((320, 320)).astype(np.float32), (0, 0), 1.5); n = n / n.std()
    Y = 0.25 * 10 ** (0.08 * n); img = np.stack([Y, Y, Y], -1).astype(np.float32); img[120:200, 120:200] *= 3
    d = enhance.MicroContrast()(img, 8.82); L = np.log10(Y * np.where((np.arange(320)[:, None] >= 120) & (np.arange(320)[:, None] < 200) & (np.arange(320)[None, :] >= 120) & (np.arange(320)[None, :] < 200), 3, 1))
    hp = lambda a: a - cv2.GaussianBlur(a.astype(np.float32), (0, 0), 40 / 8.82); a, b = hp(L)[20:-20, 20:-20], d[20:-20, 20:-20]
    assert (a * b).mean() > 0 and (a + 0.6 * b).std() > a.std(), ((a * b).mean(), (a + 0.6 * b).std(), a.std())


def test_ai_gpu_agrees_with_cpu():
    """The Iris Xe driver once returned wrong numbers with no error: an available GPU must give the CPU's detail, or MicroContrast must have refused it."""
    import cv2, enhance
    rng = np.random.default_rng(2); n = cv2.GaussianBlur(rng.standard_normal((300, 420)).astype(np.float32), (0, 0), 2); n /= n.std()
    img = (0.25 * 10 ** (0.1 * n))[..., None].repeat(3, -1).astype(np.float32); img[..., 0] *= 1.3
    cpu = enhance.MicroContrast("CPU"); gpu = enhance.MicroContrast("GPU")
    if gpu.cm is None:
        assert "failed" in gpu.backend or "openvino" in gpu.backend
        return
    a, b = cpu(img, 8.82), gpu(img, 8.82); assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.95 and abs(b.std() / a.std() - 1) < 0.1, (np.corrcoef(a.ravel(), b.ravel())[0, 1], a.std(), b.std())


def test_tonal_enhancement_keeps_flats_and_avoids_halos():
    """Same gain on all channels, nothing on flats, no halo at a big edge, about k more contrast on faint local detail."""
    import restore, cv2
    run = lambda L: np.log10(restore.apply_detail(10 ** L, restore.tone_detail(10 ** L, 8.82)))
    flat = np.full((200, 200, 3), -0.8, np.float32); assert np.abs(run(flat) - flat).max() < 1e-3
    step = np.full((400, 800, 3), -1.5, np.float32); step[:, 400:] = 0.0  # 1.5 log units: a big edge, where halos would show
    out = run(step)[200, :, 1]
    assert abs(out[20] + 1.5) < 0.02 and abs(out[780]) < 0.02  # far from the edge, nothing moves
    assert out[:400].min() > -1.5 - 0.03 and out[400:].max() < 0.03, (out[:400].min(), out[400:].max())  # no halo at the step
    rng = np.random.default_rng(0); tex = cv2.GaussianBlur(rng.standard_normal((400, 400)).astype(np.float32), (0, 0), 12)[..., None].repeat(3, -1); tex = tex / tex.std() * 0.05 - 0.8
    assert 1.15 < run(tex).std() / tex.std() < 1.4
    col = np.float32([0.5, 0.2, 0.1]) * 10 ** tex; g = restore.apply_detail(col, restore.tone_detail(col, 8.82)); assert np.abs(g[..., 0] / g[..., 2] - 5).max() < 1e-3  # colour ratios stay


def test_phone_blur_removal_keeps_level_and_sharpens():
    import acutance2, restore
    H, W = 256, 256; img = np.full((H, W, 3), 0.2, np.float32); assert np.abs(restore.remove_phone_blur(img, 8.82, acutance2.REDMI_TURBO4PRO) - img).max() < 2e-3
    img[:, 128:] = 0.6; out = restore.remove_phone_blur(img, 8.82, acutance2.REDMI_TURBO4PRO)[128, :, 1]
    assert np.diff(out[120:136]).max() > 0.4 / 3 and abs(out[40] - 0.2) < 0.01 and abs(out[216] - 0.6) < 0.02


def test_enhancement_only_for_low_bit_input():
    import route
    plan = lambda **k: route.plan(dict(source_class="raw-own", make="Xiaomi", raw=dict(bits=10), **k))["ai_enhancement"]
    assert plan() is True
    assert route.plan(dict(source_class="raw-own", make="Canon", raw=dict(bits=14)))["ai_enhancement"] is False  # enough bits
    assert route.plan(dict(source_class="raw-linear-dng", make="Xiaomi", raw=dict(bits=10)))["ai_enhancement"] is False  # stacked
    assert route.plan(dict(source_class="hdr-gainmap"))["ai_enhancement"] is False and route.plan(dict(source_class="encoded-float"))["ai_enhancement"] is False
    assert route.plan(dict(source_class="sdr"))["ai_enhancement"] is True


def test_noise_estimate_finds_known_noise():
    import camera_raw, cv2
    rng = np.random.default_rng(0); yy, xx = np.mgrid[0:1600, 0:1600] / 1600; base = (0.02 + 0.7 * xx * yy + 0.05 * yy).astype(np.float32); base = cv2.GaussianBlur(base, (0, 0), 3)
    for a, b in ((1e-4, 1e-6), (5e-4, 1e-5)):
        c = base + rng.standard_normal(base.shape, dtype=np.float32) * np.sqrt(a * base + b); e = camera_raw.estimate_noise(c, [1, 0, 2, 1])
        assert 0.85 < np.sqrt((e["a"] * 0.18 + e["b"]) / (a * 0.18 + b)) < 1.2, (a, b, e)  # sigma at 18 % grey


def test_lens_is_never_guessed():
    """A wrong lens is worse than none: a file with no lens name on an interchangeable-lens body gets no correction; a named lens does."""
    import camera_raw
    got, why = camera_raw.lens_modifier(dict(make="PENTAX", model="PENTAX K-r", focal_mm=55.0, fnumber=11.0, lens=None), 4308, 2868); assert got is None and "lens not identified" in why
    got, why = camera_raw.lens_modifier(dict(make="NIKON CORPORATION", model="NIKON Z fc", focal_mm=46.0, fnumber=9.0, lens="18-70mm f/3.5-4.5G"), 5600, 3728); assert got is not None and "18-70" in got[1], why
    got, why = camera_raw.lens_modifier(dict(make="Panasonic", model="DMC-ZS100", focal_mm=9.1, fnumber=2.8, lens=None), 5488, 3664); assert got is not None, why  # fixed lens
    got, why = camera_raw.lens_modifier(dict(make="SONY", model="ZV-E10M2", focal_mm=35.0, focal_35mm=52.0, fnumber=1.8, lens="E 35mm F1.8 OSS"), 6240, 4168); assert got is not None and "E 35mm" in got[1], why  # new body, lens known by name


def test_camera_raw_matches_libraw_colour_and_keeps_range():
    """Mosaic path against LibRaw's finished image on a real file: same colour and level where LibRaw did not clip."""
    import camera_raw, ingest, cv2
    fs = list((ROOT / "samples" / "test_formats" / "raw").glob("SONY*ARW"))
    if not fs:
        pytest.skip("sample not present")
    img, rep = camera_raw.load(fs[0], lens=False); old, _ = ingest._libraw(fs[0]); assert img.shape == old.shape and rep["raw"]["bits"] == 14
    sm = lambda x: cv2.resize(x, (x.shape[1] // 16, x.shape[0] // 16), interpolation=cv2.INTER_AREA); A, B = sm(img), sm(old); ok = (B.max(-1) < 0.9) & (B.min(-1) > 0.005); d = np.log2(A[ok] / B[ok]); assert np.abs(np.median(d, 0)).max() < 0.03, np.median(d, 0)


def test_camera_sharpness_switch_removes_the_film_scatter(k64):
    import acutance2
    st, _, _ = k64; W = 2048; pitch = 8.82; xs = np.arange(W) * pitch / 1000.0; ve = round(40 * xs[-1]) / (W * pitch / 1000.0); lh0 = float(np.interp(1.0, st.D[::-1, 1], st.loge[::-1])); L = np.zeros((12, W, 3), np.float32) + (lh0 + 0.004 * np.sin(2 * np.pi * ve * xs))[None, :, None]
    resp = lambda p: (lambda r: (r.max() - r.min()) / 0.008)(acutance2.apply(st, L, p, pitch)[0][6, W // 4:3 * W // 4, 1])
    film_, cam = resp(acutance2.KODACHROME_64), resp(dict(acutance2.KODACHROME_64, K_um=(0.0, 0.0, 0.0))); assert cam > film_ * 1.5 and cam >= 0.95, (film_, cam)  # at 40 c/mm the film alone drops well below the file


def test_every_card_has_a_sound_stock():
    """Each film on the shelf: loads, speed agrees with its printed ISO, the three layers are named sanely, labelled data, and grey stays near the middle of the curve."""
    import json, studio
    for fid, sid in studio.FILM_STOCK.items():
        path = ROOT / "data" / "stocks" / f"{sid}.cfstock.json"; st = film.Stock(path); P = json.loads(path.read_text())["primary"]
        assert all(P[k]["provenance"] in ("datasheet", "assumption") for k in P), fid
        assert fid in studio.FILM_TAG and st.rms and st.iso, fid
        assert abs(float(np.interp(1.0, st.D[::-1, 1], st.loge[::-1])) - (1.07 - np.log10(st.iso))) < 0.15, (fid, float(np.interp(1.0, st.D[::-1, 1], st.loge[::-1])))
        assert (st.D[0] > 2.5).all() and (st.D[-1] < 0.35).all() and (np.diff(st.D, axis=0) <= 1e-6).all(), fid  # a reversal film: density falls with exposure
        peaks = [int(np.argmax(st.sens[:, c])) for c in range(3)]; assert peaks[0] > peaks[1] > peaks[2], (fid, peaks)  # red, green, blue layers
        dp = [int(np.argmax(st.dye[:, c])) for c in range(3)]; assert dp[0] > dp[1] > dp[2], (fid, dp)  # cyan, magenta, yellow dyes


def test_oil_colour_keeps_greys_and_deepens_dark_colour(k64):
    """Colour depth: greys unchanged; colours purer, more so where there is more dye. Layer thickness: none on neutral areas, and brightness holds on average."""
    import oil
    st, view, _ = k64; lamp = colour.sd_blackbody(3200, film.SHAPE).values; a = film.bake_lut(st, lamp, viewing=None); b = film.bake_lut(st, lamp, viewing=None, depth=0.3); look = lambda L, l: film.apply_lut_cpu(np.array([[L]], np.float32), l, "bt2020")[0, 0]
    for g in (-1.6, -1.0, -0.5):
        assert np.abs(look([g, g, g], b) - look([g, g, g], a)).max() < 0.005, g
    sat = lambda c: (c.max() - c.min()) / max(c.max(), 1e-6); dark, light = [-0.7, -1.6, -1.6], [0.0, -0.6, -0.6]
    assert sat(look(dark, b)) > sat(look(dark, a)) and look(dark, b).max() > 0
    gain = lambda L: float(np.linalg.norm(look(L, b) - look(L, a)) / max(np.linalg.norm(look(L, a)), 1e-6)); assert gain(dark) > gain(light), (gain(dark), gain(light))
    grey = np.full((300, 300, 3), -0.8, np.float32); assert np.abs(oil.layer_thickness(grey, 8.82)).max() < 1e-5
    yy = np.linspace(0, 6.28 * 4, 300)[None, :].repeat(300, 0).astype(np.float32); col = np.stack([-0.6 + 0.2 * np.sin(yy), -1.2 - 0.2 * np.sin(yy), -1.2 * np.ones_like(yy)], -1); d = oil.layer_thickness(col, 40.0); assert np.abs(d).max() > 0.01 and np.abs(d.mean(-1)).max() < 0.1


def test_graded_cards_keep_the_film_and_have_their_own_table():
    """Same Kodachrome 64 stock under both graded cards. Their tables differ from the plain film and from each other, and HDR values above white survive the look."""
    import looks, studio
    st = studio.Studio(); st.use("k14"); a = st.lut("lightbox")["bt2020"]; base = st.stock
    for fid in ("k14legacy", "k14digital"):
        st.use(fid); assert st.stock.name == base.name and np.array_equal(st.stock.D, base.D) and np.array_equal(st.stock.dye, base.dye); assert np.abs(st.lut("lightbox")["bt2020"] - a).max() > 0.01, fid
    st.use("k14legacy"); l1 = st.lut("lightbox")["bt2020"]; st.use("k14digital"); assert np.abs(st.lut("lightbox")["bt2020"] - l1).max() > 0.01
    assert studio.film_defaults("k14legacy")["grain"] == 1.0 and studio.film_defaults("k14")["grain"] == 0.5 and studio.FILM_ACUTANCE["k14legacy"] is studio.FILM_ACUTANCE["k14"]
    st.use("provia"); plain = st.lut("lightbox")["bt2020"]; pd = studio.film_defaults("provialegacy"); assert pd["interimage"] == studio.film_defaults("provia")["interimage"] and pd["micro"] == 0.6; st.use("provialegacy"); assert np.array_equal(st.stock.D, film.Stock(ROOT / "data" / "stocks" / "provia_100f_sheet.cfstock.json").D) and np.abs(st.lut("lightbox")["bt2020"] - plain).max() > 0.01; assert studio.film_acutance("provialegacy")["A"] == studio.FILM_PROFILE["provia"]["acutance"]["A"]
    p = looks.load("kodachrome_legacy"); hi = np.array([[130.0, 0.0, 0.0], [110.0, 5.0, 5.0]], np.float32); assert (looks.grade_lab(hi, p)[..., 0] > 100).all()


def test_grain_slider_gives_twice_the_grain_on_every_card():
    import studio
    assert studio.GRAIN_GAIN == 2.0 and studio.film_defaults("k14")["grain"] == 0.5 and studio.film_defaults("k14legacy")["grain"] == 1.0
    st = studio.Studio(); sc = np.full((512, 512, 3), 0.18, np.float32)
    class S:
        pass
    for fid in ("k14", "provia"):
        st.use(fid); import acutance2, dye_clouds, grain
        L = film.scene_to_logH(sc, st.M, st.stock); upx = 8.82
        g = (lambda k: grain.apply(st.stock, L, dict(grain.KODACHROME_64, rms=st.stock.rms, strength=k), upx, st.v_k, seed=0)[0]) if fid == "k14" else (lambda k: dye_clouds.apply(st.stock, L, upx, st.v_k, strength=k, seed=0))
        ratio = float(g(2.0)[..., 1].std() / g(1.0)[..., 1].std()); assert 1.7 < ratio < 2.3, (fid, ratio)


def test_film_table_cache_returns_the_same_table(tmp_path, monkeypatch):
    import studio
    monkeypatch.setattr(studio, "LUT_CACHE", tmp_path); st = studio.Studio(); a = st._baked("kodachrome_64", st.stock, "lightbox"); b = st._baked("kodachrome_64", st.stock, "lightbox"); assert list(tmp_path.glob("*.npz")) and a["grid"] == b["grid"] and np.array_equal(a["bt2020"], b["bt2020"])


def test_each_e6_card_has_its_own_defaults():
    """Each E-6 card: interimage inside the E-6 patent window, labelled; grey near neutral with its own offset; grain at its sheet's RMS through the aperture. Kodachrome cards keep theirs."""
    import cv2, colour, dye_clouds, grain, studio
    assert studio.film_defaults("k14")["interimage"] == 0.6 and studio.film_acutance("k14") is studio.FILM_ACUTANCE["k14"] and "k14" not in studio.FILM_PROFILE
    for fid, sid in studio.FILM_STOCK.items():
        if fid in ("k14", "k25", "k200") or fid in studio.FILM_LOOK:
            continue
        prof = studio.FILM_PROFILE[fid]; assert prof["stock"] == sid and 0.66 <= prof["interimage"] <= 1.34 and "assumption" in prof["provenance"]["interimage"], fid
        st = film.Stock(ROOT / "data" / "stocks" / f"{sid}.cfstock.json"); assert st.process == "E-6"; M, _ = film.fit_exposure_matrix(st, film.daylight(5500)); assert max(abs(v) for v in prof["neutral_offset"]) < 0.2, (fid, prof["neutral_offset"])
        d = film.daylight(5000); v_k = grain.unit_visual_density(st, film.CMF[:, 1], d); upx = 8.82; a_ref = np.array([float(np.interp(1.0, st.D[::-1, 1], st.amounts[::-1, c])) for c in range(3)]); lh = np.array([np.interp(a_ref[c], st.amounts[::-1, c], st.loge[::-1]) for c in range(3)], np.float32); g = dye_clouds.apply(st, np.zeros((1024, 1024, 3), np.float32) + lh, upx, v_k, cloud_um=prof["grain"]["cloud_um"], seed=3)
        amt = np.stack([np.interp(g[..., c], st.loge, st.amounts[:, c]) for c in range(3)], -1) - a_ref; r = 24 / upx; R = int(r) + 2; yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; disc = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32); disc /= disc.sum(); rms = float(cv2.filter2D((amt @ v_k).astype(np.float32), -1, disc)[R:-R, R:-R].std() * 1000); assert abs(rms / st.rms - 1) < 0.2, (fid, rms, st.rms)


def test_kodachrome_25_and_200_datasheet_stocks():
    """Speed gaps from the digitised curves must agree with the ISO speeds printed by Kodak; every datasheet field stays labelled; MTF fit stays near Kodak."""
    if not (ROOT / "data" / "kodak_e55").exists():
        pytest.skip("digitised sheet curves are not in the public copy")
    import acutance2, json
    g = {i: film.Stock(ROOT / "data" / "stocks" / f"kodachrome_{i}.cfstock.json") for i in (25, 64, 200)}
    h = {i: float(np.interp(1.0, s.D[::-1, 1], s.loge[::-1])) for i, s in g.items()}
    assert abs((h[25] - h[64]) - np.log10(64 / 25)) < 0.03 and abs((h[64] - h[200]) - np.log10(200 / 64)) < 0.03, h
    assert (g[25].rms, g[64].rms, g[200].rms) == (9, 10, 16)
    for i in (25, 200):
        P = json.loads((ROOT / "data" / "stocks" / f"kodachrome_{i}.cfstock.json").read_text())["primary"]
        assert all(P[k]["provenance"] == "datasheet" for k in ("density_curves", "log_sensitivity", "dye_density", "mtf", "rms_granularity", "iso")) and "base_density" not in P and P["base_mix"]["provenance"] == "assumption"
    st = g[25]; p = dict(acutance2.KODACHROME_25, halation=(0, 0, 0)); W = 4096; pitch = 36000 / W; d = np.loadtxt(ROOT / "data" / "kodak_e55" / "e55-2009_06_p3_MTF_V.csv", delimiter=",", skiprows=1); d = d[np.argsort(d[:, 0])]
    lh0 = float(np.interp(p["ref_density"], st.D[::-1, 1], st.loge[::-1])); xs = np.arange(W) * pitch / 1000.0
    for v in (5, 10, 20, 40):
        k = round(v * xs[-1]); ve = k / (W * pitch / 1000.0); logH = np.zeros((12, W, 3), np.float32) + (lh0 + 0.004 * np.sin(2 * np.pi * ve * xs))[None, :, None]
        row = acutance2.apply(st, logH, p, pitch)[0][6, W // 4:3 * W // 4, 1]; assert abs((row.max() - row.min()) / 0.008 - np.interp(ve, d[:, 0], d[:, 1] / 100)) <= 0.05, v


def test_compiled_decoder_matches_reference():
    import fastops, lj92, tifffile
    dng = next((ROOT / "samples").glob("*.dng"), None)
    if dng is None:
        pytest.skip("no sample photos in the public copy")
    with tifffile.TiffFile(dng) as t:
        tg = t.pages[0].tags; t.filehandle.seek(tg["TileOffsets"].value[5]); data = t.filehandle.read(tg["TileByteCounts"].value[5])
    ref, info = lj92.decode(data); out, info2 = fastops.lj92_decode(data)
    assert info == info2 and np.array_equal(out, ref.reshape(out.shape))


def test_fast_tables_match_np_interp(k64):
    import fastops
    st, _, _ = k64; cv = fastops.curves(st); rng = np.random.default_rng(0); x = rng.uniform(-3.5, 4.5, (300, 300)).astype(np.float32)
    for c in range(3):
        a = cv.amount(x, c); assert np.abs(a - np.interp(x, st.loge, st.amounts[:, c])).max() < 1e-3
        curve = st.amounts[:, c]; am = np.linspace(curve.min() + 0.02, curve.max() - 0.02, 400, dtype=np.float32)[None]
        assert np.abs(cv.logH(am, c) - np.interp(am, curve[::-1], st.loge[::-1])).max() < 0.02  # inverse = the old reversed np.interp


def test_gpu_display_chain_matches_cpu(k64):
    gpu = pytest.importorskip("gpu"); import output, studio
    st, view, lut = k64; rng = np.random.default_rng(2); logH = rng.uniform(-2.8, 0.8, (211, 317, 3)).astype(np.float32)
    s = studio.Studio.__new__(studio.Studio); M = np.linalg.inv(output.rgb_to_xyz(output.SPACES["p3"])) @ np.linalg.inv(output.XYZ_TO_BT2020)
    for a in (dict(), dict(exposure=0.5, temp=30, tint=-20, contrast=40, low=30, mid=-40, high=50, saturation=25, vibrance=60)):
        cpu = s.develop_cpu(logH, a, lut)
        g = gpu.develop(logH, studio.adjust_offsets(a), lut["bt2020"], lut["grid"], M, a.get("low", 0), a.get("mid", 0), a.get("high", 0), a.get("contrast", 0), a.get("saturation", 0), a.get("vibrance", 0), output.gamut_scale())
        assert np.abs(g - cpu).max() < 2e-3, np.abs(g - cpu).max()


def test_gpu_matches_cpu(k64):
    gpu = pytest.importorskip("gpu")
    st, _, lut = k64
    M = np.eye(3, dtype=np.float32); rng = np.random.default_rng(1)
    img = rng.uniform(0, 2, (300, 517, 3)).astype(np.float32)
    off = float(np.log10(10 / st.iso))
    cpu = film.apply_lut_cpu(film.scene_to_logH(img, M, st), lut)
    out = gpu.film_lut(img, M, off, lut["bt709"], lut["grid"])
    assert np.abs(out - cpu).max() < 2e-4
