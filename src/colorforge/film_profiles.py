"""Each E-6 card's own physics, colour and grain defaults. Writes data/stocks/film_physics.json, which studio.py and the app read. Kodachrome cards keep their existing values.

  Sharpness   edge gain A from the sheet's MTF. Kodak prints R, G, B MTF: each layer gets its own light scatter K. Fuji prints one curve: its K is shared by the layers (`assumption`).
  Interimage  1.0 on every card: the middle of the E-6 patent window, Fuji EP 0 442 323, 0.20-0.45 logE at D 0.5 (`assumption`). No maker publishes it per film.
              Tried and rejected: choosing it from each film's IT8 colour reach. Reach hardly responds to it (colour_reach below) and never reaches a pass mark, so any per-film value would be noise.
  Colour      log-exposure offsets on R and B so a daylight grey renders neutral on this film (`assumption`: real films vary by batch and were balanced to be neutral; sheets do not give it).
  Grain       dye clouds at the sheet's RMS (dye_clouds.py). Cloud size `assumption`.
Usage: python film_profiles.py <folder with IT8 batch folders>
"""
import glob, json, re, sys, zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import acutance2, film, fit_mtf, interimage

OUT = Path(__file__).resolve().parents[2] / "data" / "stocks"
E6 = dict(velvia50=("velvia_50_sheet", "V040510"), velvia100f=("velvia_100f", "N031018"), provia=("provia_100f_sheet", "F040425"), astia=("astia_100f", None), e100=("ektachrome_e100", "E040528"), e100g=("ektachrome_e100g", "family:e100"), e100vs=("ektachrome_e100vs", "family:e100"), epn=("ektachrome_epn", "family:e100"), e100d=("ektachrome_100d_sheet", "family:e100"))
LAMP = lambda: film.daylight(5000); STRENGTHS = (0.67, 0.83, 1.0, 1.17, 1.33); CLOUD_UM = 5.0


def lab(T):
    d = LAMP(); XYZ = (T * d) @ film.CMF / (d @ film.CMF[:, 1]); wn = d @ film.CMF / (d @ film.CMF[:, 1]); t = XYZ / wn; f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116); return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def envelope(L):
    """Largest chroma in each of 12 hue sectors, ignoring single outliers."""
    a, b = L[..., 1].ravel(), L[..., 2].ravel(); C = np.hypot(a, b); h = (np.degrees(np.arctan2(b, a)) % 360 // 30).astype(int); return np.array([np.percentile(C[h == i], 97) if (h == i).sum() > 3 else 0.0 for i in range(12)])


def target_envelope(folder, name):
    z = zipfile.ZipFile(next(glob.iglob(f"{folder}/*/{name}.zip"))); t = z.read(next(n for n in z.namelist() if n.lower().endswith(".cgt"))).decode("latin-1"); fmt = re.search(r"BEGIN_DATA_FORMAT\s+(.*?)\s+END_DATA_FORMAT", t, re.S).group(1).split(); tok = re.search(r"BEGIN_DATA\s+(.*?)\s+END_DATA\b", t, re.S).group(1).replace('"', " ").split(); k = len(fmt); rows = [tok[i:i + k] for i in range(0, len(tok), k)]; nm = [i for i, c in enumerate(fmt) if c == "SPECTRAL_NM"]; pc = [i for i, c in enumerate(fmt) if c == "SPECTRAL_PCT"]; g = np.array([float(rows[0][i]) for i in nm]); wl = np.array(film.SHAPE.wavelengths)
    T = np.stack([np.interp(wl, g, [float(r[i]) / 100 for i in pc]) for r in rows]); keep = ~np.array([r[0][1:].isdigit() and int(r[0][1:]) >= 20 for r in rows]); return envelope(lab(T[keep]))


def chart(M, st):
    """Saturated scene colours at several exposures, as flat patches."""
    vals = (0.02, 0.08, 0.3, 1.0); cols = np.array([[r, g, b] for r in vals for g in vals for b in vals], np.float32) * 0.5; n = int(np.ceil(np.sqrt(len(cols)))); P = 24; im = np.zeros((n * P, n * P, 3), np.float32)
    for i, c in enumerate(cols):
        y, x = i // n, i % n; im[y * P:(y + 1) * P, x * P:(x + 1) * P] = c
    return film.scene_to_logH(im, M, st)


def reach(st, M, tgt, s):
    L = chart(M, st); L2, _ = interimage.apply(st, L, dict(interimage.KODACHROME_64, strength=s), 8.82); a = np.stack([np.interp(L2[8::24, 8::24, c].ravel(), st.loge, st.amounts[:, c]) for c in range(3)], -1); T = 10 ** -(st.base + a @ st.dye.T); env = envelope(lab(T)); ok = tgt > 20; return float(np.mean(np.minimum(env[ok] / tgt[ok], 1.0)))


def neutral(st, M):
    """R, B log-exposure offsets that make an 18 % grey render neutral through this film's dyes under its light."""
    grey = film.scene_to_logH(np.full((1, 1, 3), 0.18, np.float32), M, st)[0, 0]; G = lambda d: np.array([float(np.interp(grey[c] + d[c], st.loge, st.amounts[:, c])) for c in range(3)])
    def err(x):
        a = G((x[0], 0.0, x[1])); T = 10 ** -(st.base + st.dye @ a); w = 10 ** -st.base; L = lab(T[None])[0] - lab(w[None])[0] * [0, 1, 1]; return [L[1], L[2]]
    return [round(float(v), 4) for v in least_squares(err, [0.0, 0.0], bounds=(-0.2, 0.2)).x]


def per_layer_k(sid, A):
    """K for each layer from the sheet's own R, G, B MTF, with the shared edge gain A. None if the sheet has one curve."""
    m = json.loads((OUT / f"{sid}.cfstock.json").read_text())["primary"].get("mtf", {}); pl = m.get("per_layer")
    if not pl:
        return None
    st = film.Stock(OUT / f"{sid}.cfstock.json"); out = []
    for c, n in enumerate("RGB"):
        f, y = np.array(pl[n]["cycles_per_mm"]), np.array(pl[n]["response"]); o = np.argsort(f); t = np.interp(fit_mtf.VS, f[o], y[o])
        def resid(x):
            p = dict(fit_mtf.BASE, A=A, K_um=(x[0], x[0], x[0])); lh0 = float(np.interp(p["ref_density"], st.D[::-1, 1], st.loge[::-1])); r = []
            for v in fit_mtf.VS:
                ve = round(v * fit_mtf.XS[-1]) / (fit_mtf.W * fit_mtf.PITCH / 1000.0); L = np.zeros((8, fit_mtf.W, 3), np.float32) + (lh0 + 0.004 * np.sin(2 * np.pi * ve * fit_mtf.XS))[None, :, None]; row = acutance2.apply(st, L, p, fit_mtf.PITCH)[0][4, fit_mtf.W // 4:3 * fit_mtf.W // 4, c]; r.append((row.max() - row.min()) / 0.008)
            return np.array(r) - t
        out.append(round(float(least_squares(resid, [6.0], diff_step=0.05, bounds=([1.0], [25.0]), max_nfev=8).x[0]), 2))
    return out


def build(folder):
    fits = json.loads((OUT / "mtf_fits.json").read_text()); res = {}; family = {}
    for fid, (sid, target) in E6.items():
        st = film.Stock(OUT / f"{sid}.cfstock.json"); M, _ = film.fit_exposure_matrix(st, film.daylight(5500)); fit = fits[sid]; base = fit_mtf.BASE["K_um"]; off = neutral(st, M); s = 1.0; rc = None
        why = "middle of the E-6 patent window; no per-film data exists"
        if target and not target.startswith("family:"):  # kept as the evidence for not fitting it
            tgt = target_envelope(folder, target); rc = {str(x): round(reach(st, M, tgt, x), 3) for x in STRENGTHS}
        K = per_layer_k(sid, fit["A"]); ks = K if K else [round(k * fit["K_scale"], 2) for k in base]
        res[fid] = dict(stock=sid, interimage=s, interimage_why=why, colour_reach=rc, neutral_offset=off, acutance=dict(A=fit["A"], K_um=ks, per_layer_from_sheet=bool(K), max_mtf_error=fit["max_error"]), grain=dict(model="dye_clouds", cloud_um=CLOUD_UM), provenance=dict(interimage="assumption, middle of the window in Fuji EP 0 442 323", neutral_offset="assumption", acutance="fitted to the sheet's MTF", cloud_um="assumption", rms="datasheet")); print(fid, {k: res[fid][k] for k in ("interimage", "colour_reach", "neutral_offset")}, res[fid]["acutance"], flush=True)
    (OUT / "film_physics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    build(sys.argv[1])
