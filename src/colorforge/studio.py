"""FilmForge studio engine: the five engines behind the GUI, with per-stage caching so sliders stay interactive.

  1 Input Adapt  file -> CF-Scene + capture report + routing plan (ingest.py, route.py)              slow, cached per file
  2 Physics      scene -> film layer log exposure with interimage, acutance, grain                    ~15 s, cached per settings
  3 Color        log H -> dyes -> viewing condition (3D LUT on the GPU), gamut compression in P3      fast
  4 Calibrate    exposure, white balance, tint as log-exposure offsets; contrast, saturation, vibrance on the display  fast
  5 Output       full resolution; Ultra HDR JPEG, JPEG, JPEG XL or linear DNG

Exposure and white balance are gains on the light reaching the film, so in log H they are simple offsets and need no physics re-run.
The cached interimage and acutance were computed at the slider-free exposure: an approximation for large slider moves (assumption).
Preview and export use the same path, so what you see is what you export. Films: Kodachrome 25, 64 and 200, and E-6 slide films, from the makers' sheets.
"""
import gc, hashlib, json, os, threading, time
from collections import OrderedDict
from pathlib import Path

import colour
import cv2
import numpy as np

import acutance
import acutance2
import dye_clouds
import film
import grain
import ingest
import looks
import interimage
import output
import restore
import route
import oil
from cf_dng import srgb_preview

ROOT = Path(__file__).resolve().parents[2]
PREVIEW_SIDE = 1600
PHYSICS_DEFAULTS = dict(film="k14", lens=True, ai_micro=False, tone=False, camera_sharp=False, grain=0.5, micro=1.0, clarity=0.0, deblur=True, interimage=0.6, denoise="auto")
FILM_TAG = dict(velvia50legacy="velvia50_legacy", velvia50digital="velvia50_digital", provialegacy="provia_legacy", proviadigital="provia_digital", k14legacy="kodachrome_legacy", k14digital="kodachrome_digital", k14="kodachrome64", k25="kodachrome25", k200="kodachrome200", velvia50="velvia50", velvia100f="velvia100f", provia="provia100f", astia="astia100f", e100="ektachromee100", e100g="ektachromee100g", e100vs="ektachromee100vs", epn="ektachromeepn", e100d="ektachrome100d")
FILM_STOCK = dict(velvia50legacy="velvia_50_sheet", velvia50digital="velvia_50_sheet", provialegacy="provia_100f_sheet", proviadigital="provia_100f_sheet", k14legacy="kodachrome_64", k14digital="kodachrome_64", k14="kodachrome_64", k25="kodachrome_25", k200="kodachrome_200", velvia50="velvia_50_sheet", velvia100f="velvia_100f", provia="provia_100f_sheet", astia="astia_100f", e100="ektachrome_e100", e100g="ektachrome_e100g", e100vs="ektachrome_e100vs", epn="ektachrome_epn", e100d="ektachrome_100d_sheet")
FILM_LOOK = dict(k14legacy="kodachrome_legacy", k14digital="kodachrome_digital", velvia50legacy="velvia50_legacy", velvia50digital="velvia50_digital", provialegacy="provia_legacy", proviadigital="provia_digital")  # graded cards: a film's physics plus a fitted look (looks.py)
FILM_BASE = dict(k14legacy="k14", k14digital="k14", velvia50legacy="velvia50", velvia50digital="velvia50", provialegacy="provia", proviadigital="provia")  # the film under each graded card
LOOK_TEXTURE = dict(k14=dict(micro=0.6, grain=1.0), velvia50=dict(micro=0.6, grain=0.5), provia=dict(micro=0.6, grain=0.5))  # graded cards' starting texture, per base film. Weak fit, so `assumption`
FILM_ACUTANCE = dict(k14legacy=acutance2.KODACHROME_64, k14digital=acutance2.KODACHROME_64, k14=acutance2.KODACHROME_64, k25=acutance2.KODACHROME_25, k200=acutance2.KODACHROME_200)
_PROFILES = ROOT / "data" / "stocks" / "film_physics.json"
FILM_PROFILE = json.loads(_PROFILES.read_text()) if _PROFILES.exists() else {}  # each E-6 card's own defaults, built by film_profiles.py


def film_defaults(film_id):
    """A card's own starting slider values."""
    base = FILM_BASE.get(film_id, film_id); prof = FILM_PROFILE.get(base); tex = LOOK_TEXTURE[base] if film_id in FILM_LOOK else dict(micro=1.0, grain=0.5)
    return dict(interimage=prof["interimage"] if prof else 0.6, **tex)


def film_acutance(film_id):
    """Each film's edge model, from its own printed MTF. Kodachrome cards keep theirs."""
    if film_id not in FILM_ACUTANCE:
        a = FILM_PROFILE[FILM_BASE.get(film_id, film_id)]["acutance"]; FILM_ACUTANCE[film_id] = dict(acutance2.KODACHROME_64, A=a["A"], K_um=tuple(a["K_um"]), provenance="fitted to the sheet's MTF; other acutance values as Kodachrome 64 = assumption")
    return FILM_ACUTANCE[film_id]
    fit = (json.loads(_MTF_FITS.read_text()) if _MTF_FITS.exists() else {}).get(FILM_STOCK[film_id]); base = acutance2.KODACHROME_64
    FILM_ACUTANCE[film_id] = dict(base, A=fit["A"], K_um=tuple(k * fit["K_scale"] for k in base["K_um"]), provenance=fit["provenance"]) if fit else base
    return FILM_ACUTANCE[film_id]
ADJUST_DEFAULTS = dict(oil=0.0, oil_thick=0.0, exposure=0.0, temp=0.0, tint=0.0, contrast=0.0, low=0.0, mid=0.0, high=0.0, saturation=0.0, vibrance=0.0, view="lightbox")
VIEWS = {"lightbox": "lightbox", "scan": "projection", "projected": "screen"}  # GUI name -> film.VIEWING key
ZONES = ((-3.5, 1.6), (-0.5, 1.2), (1.2, 1.0))  # centre and width in stops from 18 % display grey: shadows, mids, highlights
P3_Y = np.array([0.2290, 0.6917, 0.0793], np.float32)
MAX_CACHE_BYTES = 6e9
GRAIN_GAIN = 2.0  # user choice: the Film grain slider gives twice its former strength everywhere. Slider 75 on Kodachrome 64 is about 1.5x its sheet RMS, what a half frame gives (24 mm against 36 mm)
PHYSICS_MEMORY = 3  # renders kept per photo, so going back to a film or setting is instant
LUT_CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ColorForge" / "lut_cache"


def tone_zones(p3, low=0.0, mid=0.0, high=0.0):
    """Zone contrast, sliders -100..100. Luminance in stops is stretched around each zone's centre with a Gaussian window, and RGB is
    scaled by the luminance ratio so hue and saturation ratios stay. Creative control applied after the film; 0 = untouched."""
    if not (low or mid or high):
        return p3
    Y = np.maximum(p3 @ P3_Y, 1e-5); L = np.log2(Y / 0.18); d = np.zeros_like(L)
    for s, (c, w) in zip((low, mid, high), ZONES):
        if s:
            d += (s / 100 * 0.5) * np.exp(-0.5 * ((L - c) / w) ** 2) * (L - c)
    return p3 * (2.0 ** d)[..., None]


def global_contrast(p3, amount=0.0):
    """Slider -100..100: luminance stops stretched around 18 % display grey by up to +-50 %; RGB scaled, so colour ratios stay."""
    if not amount:
        return p3
    Y = np.maximum(p3 @ P3_Y, 1e-5)
    return p3 * ((Y / 0.18) ** (amount / 100 * 0.5))[..., None]


def saturate(p3, saturation=0.0, vibrance=0.0):
    """Sliders -100..100, luminance kept. Saturation scales all colours equally (-100 = grey, +100 = double). Vibrance scales muted colours
    most and leaves already saturated ones nearly alone. Creative controls after the film; 0 = untouched."""
    if not (saturation or vibrance):
        return p3
    Y = (p3 @ P3_Y)[..., None]; k = np.float32(1 + saturation / 100)
    if vibrance:
        hi = p3.max(-1, keepdims=True); sat = np.clip((hi - p3.min(-1, keepdims=True)) / np.maximum(hi, 1e-5), 0, 1)
        k = k * (1 + (vibrance / 100) * (1 - sat) ** 2)
    return Y + k * (p3 - Y)


def adjust_offsets(adj):
    """Exposure (EV) and white balance sliders as per-layer log10 exposure offsets."""
    return np.log10(film.manual_white(adj.get("temp", 0.0), adj.get("tint", 0.0))) + np.float32(adj.get("exposure", 0.0) * np.log10(2))


class Session:
    def __init__(self, path):
        self.path = Path(path); self.scene = self.rep = None; self.plan = None; self.small = None; self.auto_ev = 0.0; self.exp_rep = None
        self.physics = None; self.logH = None; self.small_logH = None; self.seconds = {}; self.ai_detail = None; self.deblurred = None; self.ai_used = False; self.input_s = 0.0; self.memory = OrderedDict(); self.oil_detail = None; self.small_oil_detail = None

    def bytes(self):
        return sum(a.nbytes for a in (self.scene, self.logH, self.small, self.small_logH, self.ai_detail, self.deblurred, self.oil_detail, self.small_oil_detail) if a is not None) + sum(m["logH"].nbytes + m["small_logH"].nbytes for m in self.memory.values())


class Studio:
    def __init__(self, status=lambda **k: None):
        self.status = status; self.lock = threading.RLock(); self.sessions = OrderedDict(); self.current = None; self.gpu = None; self.dn = None; self.micro = None; self.ai_cache = {}; self.device = "starting"; self.film_data = {}; self.film = None
        t = time.time(); self.use("k14"); self.icc = output.icc_profile("p3", "ColorForge Display P3")
        self.to_p3 = np.linalg.inv(output.rgb_to_xyz(output.SPACES["p3"])) @ np.linalg.inv(output.XYZ_TO_BT2020)
        try:
            if os.environ.get("COLORFORGE_COMPUTE", "").lower() == "cpu":
                raise RuntimeError("COLORFORGE_COMPUTE=cpu")
            import gpu
            gpu.device(); self.gpu = gpu; self.device = gpu.device().cf_name
        except Exception as e:
            self.device = f"CPU ({str(e)[:50]})"
        self.setup_s = round(time.time() - t, 1)

    @staticmethod
    def _key(film_id):
        """Graded cards share a stock but have their own colour tables."""
        return film_id if film_id in FILM_LOOK else FILM_STOCK[film_id]

    def use(self, film_id):
        """Make a film's stock and tables current. Each stock is loaded once and stays loaded."""
        sid = FILM_STOCK[film_id]; key = self._key(film_id)
        if key not in self.film_data:
            self.status(message=f"Loading {sid.replace('_', ' ')}"); st = film.Stock(ROOT / "data" / "stocks" / f"{sid}.cfstock.json"); M, _ = film.fit_exposure_matrix(st, film.daylight(5500))
            self.film_data[key] = dict(stock=st, M=M, off=float(np.log10(10 / st.iso)), v_k=grain.unit_visual_density(st, film.CMF[:, 1], self._lamp(st)), lut_hdr=self._baked(sid, st, "hdr", look=FILM_LOOK.get(film_id)), luts=dict(lightbox=self._baked(sid, st, VIEWS["lightbox"], look=FILM_LOOK.get(film_id))))
        self.film = film_id

    stock = property(lambda self: self.film_data[self._key(self.film)]["stock"])
    M = property(lambda self: self.film_data[self._key(self.film)]["M"])
    off = property(lambda self: self.film_data[self._key(self.film)]["off"])
    v_k = property(lambda self: self.film_data[self._key(self.film)]["v_k"])
    lut_hdr = property(lambda self: self.film_data[self._key(self.film)]["lut_hdr"])
    luts = property(lambda self: self.film_data[self._key(self.film)]["luts"])

    @staticmethod
    def _lamp(st):
        """The viewing light the film's own sheet normalises its dyes to: 3200 K on Kodachrome's E-55, 5000 K on Kodak's E-6 sheets (taken as the same for Fuji: assumption)."""
        return colour.sd_blackbody(3200, film.SHAPE).values if st.process == "K-14" else film.daylight(5000)

    lamp = property(lambda self: self._lamp(self.stock))

    @staticmethod
    def _depth(adj):
        """Colour depth slider 0-100 to the film's path-length factor, in steps, because each step is a colour table. 100 = 0.45, the most that was not destructive when checked by eye; 50 (the middle, the default) = 0.225."""
        return round(adj.get("oil", 0.0) / 12.5) * 0.05625

    def _baked(self, sid, st, viewing, depth=0.0, look=None):
        """A film's colour table, from disk when this stock and this code already made one."""
        h = hashlib.sha1((ROOT / "data" / "stocks" / f"{sid}.cfstock.json").read_bytes() + Path(film.__file__).read_bytes() + viewing.encode() + st.process.encode() + Path(looks.__file__).read_bytes() + (((looks.OUT / f"{look}.json").read_bytes()) if look else b"")).hexdigest()[:16]; p = LUT_CACHE / f"{sid}_{look or 'film'}_{viewing}_{depth:.3f}_{h}.npz"
        try:
            with np.load(p) as z:
                return dict(grid=(float(z["grid"][0]), float(z["grid"][1]), int(z["grid"][2])), bt709=z["bt709"], bt2020=z["bt2020"])
        except (OSError, KeyError, ValueError):
            lut = film.bake_lut(st, self._lamp(st), viewing=film.VIEWING[viewing], depth=depth, look=looks.load(look) if look else None); LUT_CACHE.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".tmp.npz"); np.savez(tmp, grid=np.array(lut["grid"], float), bt709=lut["bt709"], bt2020=lut["bt2020"]); tmp.replace(p)
            return lut

    def lut(self, view, depth=0.0):
        key = view if not depth else f"{view}@{depth:.3f}"
        if key not in self.luts:
            self.luts[key] = self._baked(FILM_STOCK[self.film], self.stock, VIEWS[view], depth, FILM_LOOK.get(self.film))
        return self.luts[key]

    def _hdr_scale(self, view):
        g = np.full((1, 1, 3), self.off, np.float32); Y = lambda l: float(film.apply_lut_cpu(g, l, "bt2020")[0, 0] @ [0.2627, 0.6780, 0.0593])
        return Y(self.lut(view)) / Y(self.lut_hdr)

    def _denoiser(self):
        if self.dn is None:
            from denoise import RawDenoiser
            self.dn = RawDenoiser("AUTO"); self.dn_backend = self.dn.backend
        d = self.dn
        return lambda c, pat, S, O: d(c, pat, S, O, strength=route.denoise_strength(S, O)[0])

    @staticmethod
    def input_key(p):
        """Only the denoiser and lens correction need the file read again."""
        return f"{p['denoise']}-{p.get('lens', True)}"

    def purge(self):
        """Free the pixel data of every open photo. The models and film LUTs stay loaded, so the next photo still opens quickly."""
        self.sessions.clear(); self.ai_cache.clear(); self.current = None; gc.collect()

    # ---- engine 1
    def open(self, path, physics=None):
        path = str(Path(path).resolve()); phys = dict(PHYSICS_DEFAULTS, **(physics or {})); self.use(phys["film"])
        s = self.sessions.get(path)
        if s is None or s.physics is None or self.input_key(s.physics) != self.input_key(phys):
            s = Session(path); t = time.time(); self.status(engine="input", message="Reading the file, denoising raw data")
            s.scene, s.rep = ingest.load_scene(path, self._denoiser() if phys["denoise"] == "auto" else None, lens=phys.get("lens", True))
            s.plan = route.plan(s.rep); s.auto_ev, s.exp_rep = film.auto_exposure(s.scene, self.stock); s.ev_stock = FILM_STOCK[phys["film"]]
            H, W = s.scene.shape[:2]; k = min(1.0, PREVIEW_SIDE / max(H, W))
            s.small = cv2.resize(s.scene, (max(1, int(W * k)), max(1, int(H * k))), interpolation=cv2.INTER_AREA); s.input_s = time.time() - t; s.seconds["input"] = round(s.input_s, 1)
        self.sessions[path] = s; self.sessions.move_to_end(path); self.current = s
        if s.physics != phys:
            self.run_physics(s, phys)
        while sum(x.bytes() for x in self.sessions.values()) > MAX_CACHE_BYTES and len(self.sessions) > 1:
            self.sessions.popitem(last=False)
        return s

    def restored_scene(self, s, p, upx):
        """End of engine 1: blur removal, AI micro-contrast, tonal enhancement, in that order. The film sees the result. Pieces are cached, so switching is quick."""
        t = time.time(); sc = s.scene; s.ai_used = False; deblur = bool(p["deblur"] and s.rep.get("phone_profile") == "REDMI_TURBO4PRO")
        if deblur:
            if s.deblurred is None:
                self.status(engine="input", message="Removing lens and sensor blur"); s.deblurred = restore.remove_phone_blur(s.scene, upx, acutance2.REDMI_TURBO4PRO)
            sc = s.deblurred
        if s.plan["ai_enhancement"] and p.get("ai_micro"):
            key = (str(s.path), self.input_key(p), deblur); s.ai_detail = self.ai_cache.get(key)  # survives changing film; about 30 s on the dev laptop's Iris Xe
            if s.ai_detail is None:
                self.status(engine="input", message="AI micro-contrast, first time for this photo")
                import enhance  # loads PyTorch, so only when it is used
                if self.micro is None:
                    self.micro = enhance.MicroContrast()
                s.ai_detail = self.ai_cache[key] = self.micro(sc * np.float32(2 ** s.auto_ev), upx)
                while len(self.ai_cache) > 3:
                    self.ai_cache.pop(next(iter(self.ai_cache)))
            import enhance
            sc = restore.apply_detail(sc, np.float32(enhance.MICRO_STRENGTH) * s.ai_detail); s.ai_used = True
        if s.plan["ai_enhancement"] and p.get("tone"):
            sc = restore.apply_detail(sc, restore.tone_detail(sc, upx))
        s.seconds["input"] = round(s.input_s + time.time() - t, 1)
        return sc

    # ---- engine 2
    def run_physics(self, s, p):
        self.use(p["film"]); key = json.dumps(p, sort_keys=True)
        if key in s.memory:  # this film and these settings were rendered a moment ago
            m = s.memory[key]; s.memory.move_to_end(key); s.oil_detail = s.small_oil_detail = None; s.logH, s.small_logH, s.auto_ev, s.exp_rep, s.ev_stock, s.ai_used, s.ai_detail = m["logH"], m["small_logH"], m["auto_ev"], m["exp_rep"], m["ev_stock"], m["ai_used"], m["ai_detail"]; s.physics = dict(p); s.seconds["physics"] = 0.0
            return
        if s.ev_stock != FILM_STOCK[p["film"]]:  # each film has its own speed and range
            s.auto_ev, s.exp_rep = film.auto_exposure(s.scene, self.stock); s.ev_stock = FILM_STOCK[p["film"]]
        upx = acutance.um_per_px(s.scene.shape); sc = self.restored_scene(s, p, upx)
        t = time.time(); self.status(engine="physics", message="Interimage, film edge effect and grain")
        logH = film.scene_to_logH(sc * np.float32(2 ** s.auto_ev), self.M, self.stock); prof = FILM_PROFILE.get(FILM_BASE.get(p["film"], p["film"]))
        if prof:  # this card's own grey balance
            logH = logH + np.float32([prof["neutral_offset"][0], 0.0, prof["neutral_offset"][1]])
        if p["interimage"] > 0:
            logH, _ = interimage.apply(self.stock, logH, dict(interimage.KODACHROME_64, strength=p["interimage"]), upx)
        if p["clarity"]:  # creative: wide local contrast at 200 um, either way; 0 = pure film
            m = logH.mean(-1); logH = logH + (p["clarity"] * 0.5 * (m - cv2.GaussianBlur(m, (0, 0), 200 / upx)))[..., None]
        logH, _ = acutance2.apply(self.stock, logH, dict(film_acutance(p["film"]), strength=p["micro"], **(dict(K_um=(0.0, 0.0, 0.0)) if p.get("camera_sharp") else {})), upx)  # camera_sharp: skip the film's light scatter, keep the file's own detail
        if p["grain"] > 0 and prof:  # slider 0.5 = the sheet's RMS
            logH = dye_clouds.apply(self.stock, logH, upx, self.v_k, cloud_um=prof["grain"]["cloud_um"], strength=GRAIN_GAIN * p["grain"] / 0.5, seed=0)
        elif p["grain"] > 0:
            logH, _ = grain.apply(self.stock, logH, dict(grain.KODACHROME_64, rms=self.stock.rms, strength=GRAIN_GAIN * p["grain"]), upx, self.v_k, seed=0)
        s.oil_detail = s.small_oil_detail = None; s.logH = logH; s.physics = dict(p); h, w = s.small.shape[:2]
        s.small_logH = cv2.resize(logH, (w, h), interpolation=cv2.INTER_AREA); s.seconds["physics"] = round(time.time() - t, 1)
        s.memory[key] = dict(logH=s.logH, small_logH=s.small_logH, auto_ev=s.auto_ev, exp_rep=s.exp_rep, ev_stock=s.ev_stock, ai_used=s.ai_used, ai_detail=s.ai_detail)
        while len(s.memory) > PHYSICS_MEMORY:
            s.memory.popitem(last=False)

    # ---- engines 3 + 4
    def develop(self, logH, adj, lut=None):
        """Display-linear Display P3: calibrate sliders applied, out-of-gamut colours compressed, not yet clipped."""
        lut = lut or self.lut(adj.get("view", "lightbox"), self._depth(adj)); delta = adjust_offsets(adj); g = adj.get
        if self.gpu is not None:
            with self.lock:
                return self.gpu.develop(logH, delta, lut["bt2020"], lut["grid"], self.to_p3, g("low", 0.0), g("mid", 0.0), g("high", 0.0), g("contrast", 0.0), g("saturation", 0.0), g("vibrance", 0.0), output.gamut_scale())
        return self.develop_cpu(logH, adj, lut)

    def develop_cpu(self, logH, adj, lut=None):
        """CPU reference of the display chain, also the fallback when no GPU is available."""
        lut = lut or self.lut(adj.get("view", "lightbox"), self._depth(adj)); g = adj.get
        rgb = film.apply_lut_cpu(np.asarray(logH + adjust_offsets(adj), np.float32), lut, "bt2020")
        p3 = global_contrast(tone_zones(output.from_bt2020(rgb, "p3"), g("low", 0.0), g("mid", 0.0), g("high", 0.0)), g("contrast", 0.0))
        return output.compress_gamut(saturate(p3, g("saturation", 0.0), g("vibrance", 0.0)))

    def _jpeg(self, p3, quality=90):
        return output.encode_jpeg(np.clip(p3, 0, 1), quality, self.icc)

    def _oiled(self, s, adj, logH, region=None):
        """Layer thickness: the film's layer exposures, before the dyes. region: None = whole frame, "small" = preview, or (y0, x0, h, w)."""
        c = adj.get("oil_thick", 0.0) / 100.0 * 0.45  # 100 = the most that was not destructive when checked by eye
        if not c or not adj.get("oil"):
            return logH
        um = acutance.um_per_px(s.logH.shape)
        if region == "small":
            if s.small_oil_detail is None:
                s.small_oil_detail = oil.layer_thickness(s.small_logH, um * s.logH.shape[1] / s.small_logH.shape[1])
            return logH + np.float32(c) * s.small_oil_detail
        if s.oil_detail is None:
            s.oil_detail = oil.layer_thickness(s.logH, um)
        d = s.oil_detail if region is None else s.oil_detail[region[0]:region[0] + region[2], region[1]:region[1] + region[3]]
        return logH + np.float32(c) * d

    def view(self, adj):
        s = self.current; t = time.time(); b = self._jpeg(self.develop(self._oiled(s, adj, s.small_logH, "small"), adj)); s.seconds["color"] = round(time.time() - t, 3)
        return b

    def _before(self, img, adj):
        x = srgb_preview(np.ascontiguousarray(img) * np.float32(2 ** s_ev(self.current, adj)) * film.manual_white(adj.get("temp", 0.0), adj.get("tint", 0.0)), 0.0, img.shape[1])
        return output.encode_jpeg(x.astype(np.float32) / 255, 90, None)

    def before(self, adj):
        return self._before(self.current.small, adj)

    def crop(self, adj, x0, y0, w, h, which="film"):
        """1:1 tile from full resolution, source rectangle in pixels. Grain and fine detail are only truly visible here."""
        s = self.current; H, W = s.logH.shape[:2]; w, h = max(1, min(int(w), W, 4096)), max(1, min(int(h), H, 4096))
        x0 = int(np.clip(x0, 0, W - w)); y0 = int(np.clip(y0, 0, H - h))
        if which == "before":
            return self._before(s.scene[y0:y0 + h, x0:x0 + w], adj)
        tile = s.logH[y0:y0 + h, x0:x0 + w]; return self._jpeg(self.develop(self._oiled(s, adj, tile, (y0, x0, h, w)), adj), 92)

    # ---- engine 5
    def export(self, adj, fmt, out_dir):
        s = self.current; t = time.time(); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        self.status(engine="output", message="Developing at full resolution")
        sdr = self.develop(self._oiled(s, adj, s.logH), adj); stem = f"{s.path.stem}_{FILM_TAG[s.physics['film']]}"
        if fmt == "jpeg":
            p = out_dir / f"{stem}.jpg"; p.write_bytes(output.encode_jpeg(np.clip(sdr, 0, 1), 96, self.icc))
        elif fmt == "jxl":
            p = out_dir / f"{stem}.jxl"; output.write_jxl(np.clip(sdr, 0, 1), p)
        else:
            hdr = self.develop(self._oiled(s, adj, s.logH), adj, self._baked(FILM_STOCK[self.film], self.stock, "hdr", self._depth(adj), FILM_LOOK.get(self.film))) * np.float32(self._hdr_scale(adj.get("view", "lightbox")))
            if fmt == "dng":
                p = out_dir / f"{stem}.dng"; output.write_linear_dng(hdr, p)
            else:
                p = out_dir / f"{stem}_hdr.jpg"; output.write_hdr_jpeg(np.clip(sdr, 0, 1), hdr, p)
        mask = None
        if s.ai_used:  # AI rule: every export that used AI keeps a map of what it changed (white = largest change)
            import enhance
            (out_dir / "ai_masks").mkdir(exist_ok=True); mask = out_dir / "ai_masks" / f"{p.stem}_ai_mask.png"
            cv2.imwrite(str(mask), (np.abs(s.ai_detail) / enhance.CLIP_LOG * 255).clip(0, 255).astype(np.uint8))
        rec = dict(file=p.name, ai_mask=mask.name if mask else None, source=s.path.name, format=fmt, bytes=p.stat().st_size, shape=list(s.logH.shape), seconds=round(time.time() - t, 1), adjust=adj, physics=s.physics,
                   auto_ev=round(s.auto_ev, 2), plan=s.plan, when=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as fh:  # settings of every export, for reproducing it later
            fh.write(json.dumps(rec, default=str) + "\n")
        s.seconds["output"] = rec["seconds"]; rec["path"] = str(p)
        return rec


def s_ev(s, adj):
    return s.auto_ev + adj.get("exposure", 0.0)
