"""Physical acutance v2: per-layer optical scatter in LINEAR exposure + nonlinear Nelson adjacency on first-developer silver.

1. Optics (exposure, linear light): E_k' = PSF_k * E_k. PSF_k: exponential line spread, MTF 1/(1+(2 pi K_k v)^2) (Frieser),
   plus for the bottom (red/cyan) layer a weak wide halation term h * Gaussian(sigma_h). Bright areas spill into dark ones.
2. First developer (silver fraction s = 1 - a/a_max): Nelson chemical spread, nonlinear in s
       s_eff = s + beta * s * (s - b * s),   b = w G(s_dev) + (1-w) G(s_br)       (Nelson 1971 form, as in the user's docx)
   The effect grows with local silver and with step size -> local contrast that a linear filter cannot make.
3. Back to log H through the stock's own dye-amount curve (positive dye = a_max (1 - s_eff)).

Small-signal check: around mean silver s_ref the chain gives MTF = MTF_opt(v) * [1 + beta s_ref (1 - B(v))], i.e. the linear
model fitted to Kodak's KR 64 MTF (acutance.KODACHROME_64) when beta = A / s_ref. s_ref = silver at visual-ish density 1.0
of the green layer (assumption: Kodak does not state the density of its MTF measurement).
Per-layer K and halation are assumptions (datasheet gives one visual MTF); green carries the fitted K.
"""
import cv2
import numpy as np
from scipy import fft

import fastops

KODACHROME_64 = dict(K_um=(7.54 * 1.15, 7.54, 7.54 * 0.85),  # R (bottom), G, B (top); small ratios: 35 K-14 scans show R/G/B edge widths equal within ~1 um
                     halation=(0.01, 0.0, 0.0), halation_um=150.0,  # rem-jet backing: weak; 2 % red-layer halo = assumption
                     A=1.00, s_dev=11.2, s_br=48.2, w=0.68,  # fitted to Kodak E-88 2009 KR 64 MTF (A fixed at 1)
                     ref_density=1.0, strength=1.0,
                     provenance="small-signal MTF = Kodak KR 64 fit; per-layer K ratios, halation, s_ref density = assumption")


# Same model fitted to Kodak's KM 25 MTF (E-55 06/2009 p3, max error 0.036 up to 45 c/mm); KL 200 is printed with KR 64's MTF and needs no change (max error 0.022).
KODACHROME_25 = dict(KODACHROME_64, K_um=tuple(k * 1.037 for k in KODACHROME_64["K_um"]), A=1.494, provenance="small-signal MTF = Kodak KM 25 fit; other values as Kodachrome 64 = assumption")
KODACHROME_200 = dict(KODACHROME_64, provenance="Kodak prints KL 200 with the same MTF drawing as KR 64; other values as Kodachrome 64 = assumption")

# Redmi Turbo 4 Pro main camera, measured from natural edges in its DNGs ( sharpest photo, denoise 0.65):
# MTF50 in cycles/pixel, Gaussian shape M(v) = exp(-ln2 (v/v50)^2) matches the measured curves. nsr caps the boost (assumption).
REDMI_TURBO4PRO = dict(mtf50_cy_px=(0.18, 0.26, 0.14), nsr=0.03,
                       provenance="measured on natural edges (lower bound); Wiener nsr = assumption")


def phone_wiener(f_cyc_um, pitch_um, k, phone):
    """Wiener inverse of the phone MTF for layer k: M / (M^2 + nsr), normalised to 1 at zero frequency."""
    v = f_cyc_um * pitch_um  # cycles per pixel
    M = np.exp(-np.log(2) * (v / phone["mtf50_cy_px"][k]) ** 2)
    return (M / (M ** 2 + phone["nsr"])) * (1 + phone["nsr"])


def _radial(shape, pitch_um):
    fy = fft.fftfreq(shape[0], d=pitch_um / 1000)[:, None]; fx = fft.rfftfreq(shape[1], d=pitch_um / 1000)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2) / 1000.0  # cycles per um


def _filt(x, T, pad, shape):
    xp = cv2.copyMakeBorder(x, pad, shape[0] - x.shape[0] - pad, pad, shape[1] - x.shape[1] - pad, cv2.BORDER_REFLECT)
    return fft.irfft2(fft.rfft2(xp, workers=-1) * T, s=shape, workers=-1)[pad:pad + x.shape[0], pad:pad + x.shape[1]]


def apply(stock, logH, prm, pitch_um, phone=None):
    H, W, _ = logH.shape; pad = int(np.ceil(3 * max(prm["s_br"], prm["halation_um"]) / pitch_um)); logH = np.asarray(logH, np.float32)
    shape = (fft.next_fast_len(H + 2 * pad, real=True), fft.next_fast_len(W + 2 * pad, real=True)); f = _radial(shape, pitch_um).astype(np.float32); f2 = f ** 2
    B = np.float32(prm["w"]) * np.exp(-2 * np.pi ** 2 * prm["s_dev"] ** 2 * f2) + np.float32(1 - prm["w"]) * np.exp(-2 * np.pi ** 2 * prm["s_br"] ** 2 * f2)
    cv = fastops.curves(stock); logH_ref = float(np.interp(prm["ref_density"], stock.D[::-1, 1], stock.loge[::-1]))
    out = np.empty_like(logH); info = {}
    for k in range(3):
        # 1. optics in linear exposure
        T = 1 / (1 + (2 * np.pi * prm["K_um"][k]) ** 2 * f2)
        if prm["halation"][k] > 0:
            T = (1 - prm["halation"][k]) * T + prm["halation"][k] * np.exp(-2 * np.pi ** 2 * prm["halation_um"] ** 2 * f2)
        if phone is not None:  # remove the phone's own blur first, so the image carries only the film's
            T = T * phone_wiener(f, pitch_um, k, phone).astype(np.float32)
        E = _filt(np.exp(logH[..., k] * np.float32(np.log(10))), T.astype(np.float32), pad, shape)
        lh = np.log10(np.maximum(E, 1e-8))
        # 2. first-developer silver and nonlinear Nelson adjacency
        amax = cv.amax[k]; s = 1 - cv.amount(lh, k) / np.float32(amax)
        s_ref = 1 - float(np.interp(logH_ref, stock.loge, stock.amounts[:, k])) / amax
        beta = prm["A"] / max(s_ref, 1e-3) * prm["strength"]
        s_eff = np.clip(s + np.float32(beta) * s * (s - _filt(s, B, pad, shape)), 0, 1)
        # 3. back to log H through the dye-amount curve
        out[..., k] = cv.logH(np.float32(amax) * (1 - s_eff), k)
        info["RGB"[k]] = dict(beta=round(beta, 3), s_ref=round(s_ref, 3), K_um=round(prm["K_um"][k], 2))
    return out, info
