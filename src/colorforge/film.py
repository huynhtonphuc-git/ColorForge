"""ColorForge film stage, colour part (blueprint v2 D1, D4, D6), CPU reference + LUT bake.

D1  scene BT.2020 (D65-adapted) -> film layer exposures H (lux-s) via a 3x3 fitted over 190 natural reflectances
    lit by daylight and integrated with the stock's spectral sensitivities.
D4  per-dye ("analytical") curves derived once from the datasheet's integral Status A curves (neutral series),
    then spectral density = base + sum(dye_c * amount_c(log H_c)).            [no interimage yet]
D6  transmittance x projector lamp -> XYZ, adapted so clear film (Dmin) = display white (D65).
Everything after D1 is a smooth function of (log H_R, log H_G, log H_B) and is baked into a 3D LUT.
Provenance of every model choice is in ASSUMPTIONS and goes into the render report.
"""
import json
from pathlib import Path

import colour
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

import looks
from densitometry import WL, fill_edges, status_a

SHAPE = colour.SpectralShape(380, 780, 5)
CMF = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SHAPE).values
XYZ_TO_BT2020 = np.array([[1.7166512, -0.3556708, -0.2533663], [-0.6666844, 1.6164812, 0.0157685], [0.0176399, -0.0427706, 0.9421031]])
XYZ_TO_BT709 = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
D65 = np.array([0.95047, 1.0, 1.08883])

ASSUMPTIONS = {
    "taking_illuminant": "D55 daylight (Kodak 'Daylight' has no stated CCT)",
    "viewing_illuminant": "3200 K blackbody projector lamp (Kodak normalises KR 64 dyes for 3200 K)",
    "base_density": "without a stored base: the mix of the film's own dyes that gives its printed minimum density",
    "dye_edges": "Kodak dyes drawn 420-700 nm; flat extension outside",
    "curve_ends": "datasheet curves clamped beyond the drawn range (no true Dmax drawn)",
    "grey_exposure": "18 % grey -> log H = log10(10/ISO) lux-s (ISO arithmetic speed point; to be checked vs ISO 2240)",
    "interimage": "none yet (layers independent)",
    "display_white": "clear film (Dmin) under the lamp = display white, Bradford to D65",
    "viewing": "'screen': 0.5 % projection flare, CAM16 dark surround 50 cd/m2 -> sRGB dim surround 80 cd/m2; 'projection': none",
}


def _arr(v):
    return np.array([[np.nan if x is None else x for x in r] for r in v], float) if isinstance(v[0], list) else np.array([np.nan if x is None else x for x in v], float)


class Stock:
    def __init__(self, path):
        st = json.loads(Path(path).read_text()); P = st["primary"]; self.name = st["name"]; self.process = st.get("process", "K-14")
        self.loge = np.array(st["grids"]["log_exposure"])
        self.iso = P["iso"]["value"]; self.rms = P["rms_granularity"]["value"]
        ls = _arr(P["log_sensitivity"]["value"])
        self.sens = np.where(np.isnan(ls), 0.0, 10 ** np.nan_to_num(ls, nan=-99))  # outside drawn range: no sensitivity
        self.dye = fill_edges(_arr(P["dye_density"]["value"]))
        self.dye = self.dye / status_a(self.dye.T).diagonal()  # unit dyes: Status A 1.0 in their own channel
        D = _arr(P["density_curves"]["value"])
        self.D = np.stack([np.interp(self.loge, self.loge[~np.isnan(D[:, c])], D[~np.isnan(D[:, c]), c]) for c in range(3)], 1)  # clamp ends
        self.D = np.minimum.accumulate(self.D, axis=0)  # a reversal film's density only falls with exposure: removes digitising wobble, which the inverse lookups cannot take
        dmin = self.D[-1]
        if "base_density" in P:
            base = _arr(P["base_density"]["value"]); s = least_squares(lambda p: status_a(base * p[0] + p[1]) - dmin, [1, 0]).x; self.base = base * s[0] + s[1]
        else:  # no sheet prints it: the mix of this film's own dyes that gives its printed Dmin (assumption)
            x = least_squares(lambda c: status_a(self.dye @ c) - dmin, [0.1, 0.1, 0.1], bounds=(0, 2)).x; self.base = self.dye @ x
        self.amounts = self._analytical()

    def _analytical(self):
        """Integral Status A (neutral exposure series) -> dye amounts per log H, base removed."""
        out = np.zeros_like(self.D); c = np.zeros(3)
        for i in range(len(self.loge) - 1, -1, -1):  # from Dmin upward, warm-starting each solve
            c = least_squares(lambda x: status_a(self.base + self.dye @ x) - self.D[i], c, bounds=(-0.2, 10)).x
            out[i] = c
        return out


def daylight(cct=5500):
    return colour.sd_CIE_illuminant_D_series(colour.temperature.CCT_to_xy_CIE_D(cct * 1.4388 / 1.4380)).align(SHAPE).values


def fit_exposure_matrix(stock, illum):
    """Least-squares 3x3: scene BT.2020 (white = D65) -> relative layer exposure (grey = 1)."""
    R = colour.characterisation.read_training_data_rawtoaces_v1().copy().align(SHAPE).values  # (81, 190)
    R = np.c_[R, np.ones(len(WL)) * 0.18]  # include a neutral grey
    XYZ = (R * illum[:, None]).T @ CMF
    wp = (illum @ CMF) / (illum @ CMF)[1]
    XYZ_65 = colour.adaptation.chromatic_adaptation_VonKries(XYZ / (illum @ CMF)[1], wp, D65, "Bradford")
    rgb = XYZ_65 @ XYZ_TO_BT2020.T
    E = (R * illum[:, None]).T @ stock.sens
    E = E / E[-1] * 0.18  # grey card -> 0.18 in every layer
    w = 1 / np.maximum(E, 1e-3)  # relative (log-like) error weighting
    M = np.stack([np.linalg.lstsq(rgb * w[:, [c]], E[:, c] * w[:, c], rcond=None)[0] for c in range(3)])
    err = np.log10(np.maximum(rgb @ M.T, 1e-4) / E)
    return M, dict(n_spectra=len(E), log_exposure_rms=np.sqrt((err ** 2).mean(0)).round(4).tolist(),
                   log_exposure_p95=np.percentile(np.abs(err), 95, axis=0).round(4).tolist())


# Viewing conditions (assumptions): projected slide in a dark room vs an sRGB monitor (IEC 61966-2-1: 80 cd/m2, dim).
VIEWING = {
    "projection": None,  # colorimetry of the slide on the screen, no appearance transform
    "screen": dict(flare=0.005, src=dict(L_w=50.0, surround="Dark"), dst=dict(L_w=80.0, surround="Dim")),
    # user decision 2026-11-03: reference = slide on an ISO 3664 transparency viewer (1270 cd/m2, masked = dim surround);
    # match colourfulness M (keeps the Hunt effect of the bright viewer); 0.2 % flare (assumption)
    "lightbox": dict(flare=0.002, src=dict(L_w=1270.0, surround="Dim"), dst=dict(L_w=80.0, surround="Dim"), match="C"),  # chroma held: M-match was ~2x too colourful vs 29 scans
    # HDR rendition: the slide's own colorimetry (full slide contrast), only flare; scaled later so grey matches SDR
    "hdr": dict(flare=0.002),
}


def appearance(XYZ, v):
    """Projection flare, then CAM16 appearance match: dark-surround projection -> display surround.
    XYZ relative, D65-adapted, white Y = 1. Returns XYZ for the display, white Y = 1."""
    if v is None:
        return XYZ
    f = v["flare"]; XYZ = (XYZ + f * D65) / (1 + f)
    if "src" not in v:
        return XYZ
    S = colour.appearance.VIEWING_CONDITIONS_CAM16
    sh = XYZ.shape; X = np.clip(XYZ.reshape(-1, 3), 1e-6, None) * 100
    spec = colour.XYZ_to_CAM16(X, D65 * 100, v["src"]["L_w"] * 0.2, 20, S[v["src"]["surround"]])
    if v.get("match") == "M":  # hold colourfulness, not chroma: the inverse then derives C from M at the display level
        spec = colour.appearance.CAM_Specification_CAM16(J=spec.J, M=spec.M, h=spec.h)
    out = colour.CAM16_to_XYZ(spec, D65 * 100, v["dst"]["L_w"] * 0.2, 20, S[v["dst"]["surround"]]) / 100
    out = np.where(np.isfinite(out), out, X / 100)
    return out.reshape(sh)


def bake_lut(stock, view, n=65, lo=-3.2, hi=1.0, viewing=None, depth=0.0, look=None):
    """3D LUT over (log H_R, log H_G, log H_B) -> display-referred linear BT.709 and BT.2020 (Dmin = white)."""
    g = np.linspace(lo, hi, n)
    amt = np.stack([np.interp(g, stock.loge, stock.amounts[:, c]) for c in range(3)], 1)  # (n, 3)
    # spectral density for every node: base + C(H_R) + M(H_G) + Y(H_B)
    sd = (stock.base[None, None, None, :] + amt[:, None, None, 0, None] * stock.dye[:, 0] + amt[None, :, None, 1, None] * stock.dye[:, 1]
          + amt[None, None, :, 2, None] * stock.dye[:, 2])
    if depth:
        sd = deepen(sd, view, depth, np.arange(n))
    XYZ = (10 ** -sd * view) @ CMF
    white = (10 ** -stock.base * view) @ CMF
    XYZ = XYZ / white[1]; white = white / white[1]
    XYZ = colour.adaptation.chromatic_adaptation_VonKries(XYZ, white, D65, "Bradford")
    XYZ = appearance(XYZ, viewing)
    if look:  # a card's fitted grade, after the film (looks.py); the film's own curves and dyes are untouched
        XYZ = looks.from_lab(looks.grade_lab(looks.to_lab(XYZ), look))
    return dict(grid=(lo, hi, n), bt709=(XYZ @ XYZ_TO_BT709.T).astype(np.float32), bt2020=(XYZ @ XYZ_TO_BT2020.T).astype(np.float32))


def deepen(sd, view, depth, idx):
    """Colour depth, an artistic control after Beer-Lambert: a longer light path through the dyes makes their colour purer. Each spectrum moves away from this film's own neutral of the same
    lightness, by `depth`, weighted by how much dye there is, so dark colours deepen most and light ones little. The film's greys stay where they are. Not film data (`assumption`)."""
    w = view @ CMF[:, 1]; vis = lambda d: -np.log10(np.maximum((10 ** -d * view) @ CMF[:, 1] / w, 1e-9)); diag = sd[idx, idx, idx]; dv = vis(diag); o = np.argsort(dv); v = vis(sd); out = sd.copy()
    for k in range(sd.shape[-1]):
        out[..., k] = sd[..., k] + depth * np.clip(v / 0.8, 0.15, 1.0) * (sd[..., k] - np.interp(v, dv[o], diag[o, k]))
    return np.maximum(out, 0.0)


def grey_balance(lut, logH_grey, key="bt2020", target_ab=(0.8, -0.5)):
    """Lab-style filter pack: log-exposure offsets (R, B) so an 18 % grey lands on target a*, b*. The target is the median neutral
    balance measured on 29 Kodachrome scans (it varies by scanner, so it is approximate)."""
    wp = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]; inv = np.linalg.inv(XYZ_TO_BT2020).T

    def res(p):
        lh = np.array([[[logH_grey + p[0], logH_grey, logH_grey + p[1]]]], np.float32)
        L = colour.XYZ_to_Lab(apply_lut_cpu(lh, lut, key)[0, 0] @ inv, wp)
        return [L[1] - target_ab[0], L[2] - target_ab[1]]
    return least_squares(res, [0.02, -0.02], diff_step=0.05).x  # non-zero start: diff_step is relative to x


def auto_white(img, cct=None, strength=0.8, max_stops=0.5, daylight_k=5000):
    """Residual white correction on top of the camera's white balance.
    Daylight as-shot (>= daylight_k): trust the camera (blue sky and foliage must not be read as a cast).
    Warm light or unknown CCT: Minkowski (p=6) mean of pixels already close to the neutral axis, 1/8 size, gains limited; unknown CCT at
    half strength. Thresholds are choices, not measurements."""
    if cct is not None and cct >= daylight_k:
        return np.ones(3, np.float32), dict(gains_rgb=[1.0, 1.0, 1.0], reason=f"as-shot {cct} K: camera trusted")
    x = img[::8, ::8].reshape(-1, 3); x = x[(x > 1e-4).all(1)]
    Y = x @ np.array([0.2627, 0.6780, 0.0593], np.float32); lo, hi = np.percentile(Y, [20, 98]); x = x[(Y > lo) & (Y < hi)]
    near = (np.abs(np.log(x[:, 0] / x[:, 1])) < 0.3) & (np.abs(np.log(x[:, 2] / x[:, 1])) < 0.3)
    if near.mean() < 0.03:
        return np.ones(3, np.float32), dict(gains_rgb=[1.0, 1.0, 1.0], reason="too few near-neutral pixels")
    e = np.mean(x[near].astype(np.float64) ** 6, 0) ** (1 / 6)
    st = strength if cct is not None else strength / 2
    gains = np.clip(e[1] / e, 2.0 ** -max_stops, 2.0 ** max_stops) ** st; gains = gains / gains[1]
    return gains.astype(np.float32), dict(gains_rgb=gains.round(3).tolist(), near_neutral_frac=round(float(near.mean()), 3), cct=cct, reason="warm or unknown light")


def manual_white(temp=0.0, tint=0.0):
    """User sliders on top of the camera's white balance, scene-linear channel gains. temp: + warmer, - cooler; tint: + magenta, - green.
    Range -100..100, where 100 = 1 stop between the two ends. BT.2020 luminance is kept."""
    g = np.array([2.0 ** (temp / 200), 2.0 ** (-tint / 200), 2.0 ** (-temp / 200)])
    return (g / (g @ [0.2627, 0.6780, 0.0593])).astype(np.float32)


def auto_exposure(img, stock, highlight_pct=98.0, max_sacrifice_ev=0.75, slide_bias_ev=-0.5):
    """Per-photo exposure like a slide photographer, read from this photo's own luminance:
    - key: geometric mean of scene luminance -> 18 % grey (average metering in log space)
    - highlight: at most (100 - highlight_pct) % of the frame may reach clear film: the p98 luminance (1/8-size image)
      -> the film exposure where green density = Dmin + 0.03 (from the stock's own curve). Small lamps/windows may blow.
    The key target carries a slide bias once (real K-14 scans are denser: median L* ~35). Take the darker of key and highlight
    (protect highlights), but never more than max_sacrifice_ev below the key, so backlit subjects are not crushed."""
    small = img[::8, ::8].reshape(-1, 3) @ np.array([0.2627, 0.6780, 0.0593], np.float32)
    Y = small[small > 1e-5]
    ev_key = float(np.log2(0.18 / np.exp(np.mean(np.log(Y))))) + slide_bias_ev
    g = stock.D[:, 1]; d_t = g[-1] + 0.03
    logH_t = float(np.interp(d_t, g[::-1], stock.loge[::-1]))
    s_t = 0.18 * 10 ** (logH_t - np.log10(10 / stock.iso))
    ev_hi = float(np.log2(s_t / np.percentile(Y, highlight_pct)))
    ev = float(np.clip(max(min(ev_key, ev_hi), ev_key - max_sacrifice_ev), -6, 6))
    return ev, dict(ev=round(ev, 2), ev_key=round(ev_key, 2), ev_highlight=round(ev_hi, 2),
                    limited_by="highlights" if ev_hi < ev_key else "key", highlight_target_logH=round(logH_t, 3))


def scene_to_logH(img, M, stock, ev=0.0):
    """D1: scene-linear BT.2020 -> log10 exposure (lux-s). 18 % grey -> log10(10/ISO) + ev."""
    E = np.maximum(img.reshape(-1, 3) @ M.T.astype(np.float32), 1e-6) / 0.18
    return (np.log10(E) + np.log10(10 / stock.iso) + ev * np.log10(2)).astype(np.float32).reshape(img.shape)


def apply_lut_cpu(logH, lut, key="bt709"):
    lo, hi, n = lut["grid"]; idx = (np.clip(logH, lo, hi) - lo) / (hi - lo) * (n - 1)
    flat = idx.reshape(-1, 3).T
    out = np.stack([map_coordinates(lut[key][..., c], flat, order=1, mode="nearest") for c in range(3)], 1)
    return out.reshape(logH.shape).astype(np.float32)


def render_exact(logH, stock, view):
    """Exact per-pixel path (for tests on small arrays): log H -> amounts -> spectrum -> display BT.709."""
    amt = np.stack([np.interp(logH[..., c].ravel(), stock.loge, stock.amounts[:, c]) for c in range(3)], 1)
    sd = stock.base + amt @ stock.dye.T
    XYZ = (10 ** -sd * view) @ CMF; white = (10 ** -stock.base * view) @ CMF
    XYZ = colour.adaptation.chromatic_adaptation_VonKries(XYZ / white[1], white / white[1], D65, "Bradford")
    return (XYZ @ XYZ_TO_BT709.T).reshape(logH.shape), sd
