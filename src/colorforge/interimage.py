"""Nonlinear spatial interimage (interlayer) effect in log-exposure space, neutral-compensated.

Physics: in the reversal first developer, silver developing in one layer releases restrainer (iodide/bromide) that
diffuses into the other layers and holds their development back (Shuto 1997; Kodak US 4,082,553). Grey-scale
datasheet curves already contain this, so the model only acts on DEVIATIONS from a neutral exposure:

  s_k      = first-developer silver fraction of layer k = 1 - a_k(logH_k) / a_k,max   (a = dye amount)
  g(s)     = s^p                       highlight-weighted (silver is high where the positive is light)
  dlogH_r  = sum_{k != r} c_rk * [ g(s_k^neutral(logH_r)) - (G_sigma * g(s_k))(x) ]
  logH_r'  = logH_r + dlogH_r

Uniform grey: s_k = s_k^neutral(logH_r) -> dlogH = 0, so greys stay exactly on the Kodak curves.
Single-colour exposure (other layers unexposed, s_k = 0): dlogH_r = sum_k c_rk g(s_k^neutral) > 0, i.e. the layer
develops more silver than in neutral -> less unwanted dye -> cleaner, more saturated colour.

Calibration (assumption): c_rk chosen so the single-colour shift at analytical density 0.5 equals the middle of the
only published reversal-film ranges, Fuji EP 0 442 323 (red 0.20-0.40, green 0.25-0.45 logE at D 0.5; 0-0.07 and
0-0.15 at D 1.5). Blue has no published range: same as red (assumption). Split equally between the two donor layers.
p = 4 gives a D0.5/D1.5 ratio of ~5 (patent implies ~3-10). sigma = 20 um (our assumption; no measured value for any slide film). Kodachrome-specific values are NOT available; Q60K3 cannot identify them (unknown exposures).
"""
import cv2
import numpy as np

import fastops

KODACHROME_64 = dict(target_shift_D05=(0.30, 0.35, 0.30), p=4.0, sigma_um=20.0, strength=1.0,
                     provenance="assumption: shape and magnitude bounded by Fuji EP0442323 (E-6); no Kodachrome data exists")


def silver_fraction(stock, logH, c):
    a = np.interp(logH, stock.loge, stock.amounts[:, c]); amax = stock.amounts[:, c].max()
    return np.clip(1 - a / amax, 0, 1)


def calibrate(stock, prm):
    """c[r, k] so that sum_k c_rk g(s_k^neutral) = target shift at the logH where layer r has analytical density 0.5."""
    C = np.zeros((3, 3))
    for r in range(3):
        a = stock.amounts[:, r]; logH05 = float(np.interp(0.5, a[::-1], stock.loge[::-1]))
        gk = [silver_fraction(stock, logH05, k) ** prm["p"] for k in range(3) if k != r]
        for k, gv in zip([k for k in range(3) if k != r], gk):
            C[r, k] = prm["target_shift_D05"][r] / (2 * gv)  # equal split between donors
    return C


def apply(stock, logH, prm, um_per_px):
    """logH (H, W, 3) -> logH with interimage. Returns new logH and the per-layer shift statistics."""
    C = calibrate(stock, prm) * prm["strength"]; p = prm["p"]; sig = prm["sigma_um"] / um_per_px; cv = fastops.curves(stock); logH = np.asarray(logH, np.float32)
    # silver fraction ** p of layer k as a function of log H, tabulated once per layer
    gtab = [np.clip(1 - tab / amax, 0, 1) ** p for tab, amax in zip(cv.a, cv.amax)]
    look = lambda x, k: fastops.table(x, cv.x0, cv.step, gtab[k])
    g = [cv2.GaussianBlur(look(logH[..., k], k), (0, 0), sig, borderType=cv2.BORDER_REFLECT) for k in range(3)]
    out = logH.copy()
    for r in range(3):
        for k in range(3):
            if C[r, k]:  # what layer k holds in a grey of layer r's exposure, minus what it actually holds nearby
                out[..., r] += np.float32(C[r, k]) * (look(logH[..., r], k) - g[k])
    d = (out - logH)[::4, ::4]
    return out, dict(C=C.round(4).tolist(), shift_p1=np.percentile(d, 1, axis=(0, 1)).round(3).tolist(), shift_p99=np.percentile(d, 99, axis=(0, 1)).round(3).tolist())


def single_colour_shift(stock, prm, D):
    """Model check: shift of a single-colour exposure at analytical density D (compare with the Fuji ranges)."""
    C = calibrate(stock, prm) * prm["strength"]; out = []
    for r in range(3):
        a = stock.amounts[:, r]; lh = float(np.interp(D, a[::-1], stock.loge[::-1]))
        out.append(sum(C[r, k] * silver_fraction(stock, lh, k) ** prm["p"] for k in range(3) if k != r))
    return np.round(out, 3).tolist()
