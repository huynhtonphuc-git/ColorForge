"""Fit the film edge model to each stock's own printed MTF: edge gain A and a scale on light scatter K, up to 45 c/mm. Writes data/stocks/mtf_fits.json, which studio.py reads. Run after adding a stock.
Usage: python fit_mtf.py stock_id ...
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import acutance2, film

OUT = Path(__file__).resolve().parents[2] / "data" / "stocks"; W = 4096; PITCH = 36000 / W; XS = np.arange(W) * PITCH / 1000.0; BASE = dict(acutance2.KODACHROME_64, halation=(0, 0, 0)); VS = np.geomspace(3, 45, 10)


def curve(st, p):
    lh0 = float(np.interp(p["ref_density"], st.D[::-1, 1], st.loge[::-1])); r = []
    for v in VS:
        ve = round(v * XS[-1]) / (W * PITCH / 1000.0); L = np.zeros((8, W, 3), np.float32) + (lh0 + 0.004 * np.sin(2 * np.pi * ve * XS))[None, :, None]; row = acutance2.apply(st, L, p, PITCH)[0][4, W // 4:3 * W // 4, 1]; r.append((row.max() - row.min()) / 0.008)
    return np.array(r)


def fit(sid):
    st = film.Stock(OUT / f"{sid}.cfstock.json"); m = json.loads((OUT / f"{sid}.cfstock.json").read_text())["primary"].get("mtf")
    if not m or not m.get("value"):
        return None
    c = m["value"][0]; f, y = np.array(c["cycles_per_mm"], float), np.array(c["response"], float); o = np.argsort(f); t = np.interp(VS, f[o], y[o]); fun = lambda x: curve(st, dict(BASE, A=x[0], K_um=tuple(k * x[1] for k in BASE["K_um"]))) - t; r = least_squares(fun, [1.0, 1.0], diff_step=0.03, bounds=([0.0, 0.3], [4.0, 3.0]), max_nfev=10)
    return dict(A=round(float(r.x[0]), 3), K_scale=round(float(r.x[1]), 3), max_error=round(float(np.abs(r.fun).max()), 3), source=m.get("source", "")[:120], provenance="fitted to the printed MTF; other acutance values as Kodachrome 64 = assumption")


if __name__ == "__main__":
    p = OUT / "mtf_fits.json"; fits = json.loads(p.read_text()) if p.exists() else {}
    for sid in sys.argv[1:]:
        fits[sid] = fit(sid); print(sid, fits[sid], flush=True); p.write_text(json.dumps(fits, indent=1))
