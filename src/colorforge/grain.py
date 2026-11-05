"""Physical dye-cloud grain per layer, applied to dye amounts (blueprint v2 D5).

Each dye layer is a random set of small dye clouds (Poisson). For clouds much smaller than the sampling area the
density fluctuation follows Selwyn/Nutting: sigma_D^2 * A = const, and sigma_D^2 grows ~ linearly with density.
Calibration (datasheet): Kodak E-55 2009, diffuse RMS granularity 10 (sigma_D = 0.010) at gross diffuse visual
density 1.0, 48 um circular aperture. Assumptions: the three layers are independent with equal amplitude at unit dye
amount; their visual contributions follow each unit dye's visual (Y) density under the viewing lamp; sigma ~ sqrt(a);
dye-cloud radius 3 um (no Kodachrome measurement found) -> sub-pixel at phone resolution.
Grain is resolution-correct: per-pixel sigma scales with the pixel area on the 36 mm frame.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

import fastops

KODACHROME_64 = dict(rms=10, aperture_um=48.0, cloud_um=3.0, strength=0.5, layer_corr=0.0, density_exp=1.0,
                     provenance="datasheet RMS 10 (Kodak E-55 2009); per-layer split, sqrt(D) law, cloud size = assumption; "
                                "strength 0.5 calibrated to 29 Kodachrome scans; layer_corr 0 (independent layers: RGB grain correlation in JPEGs is dominated by chroma subsampling, not diagnostic) (research/analysis, 2026-11-03)")


def unit_visual_density(stock, cmf_y, view):
    """Visual (Y-weighted) density of one unit of each dye under the viewing lamp."""
    T0 = (view * cmf_y).sum()
    return np.array([-np.log10((10 ** -stock.dye[:, k] * view * cmf_y).sum() / T0) for k in range(3)])


def apply(stock, logH, prm, um_per_px, v_k, seed=0):
    """logH (H,W,3) -> logH with grain: logH -> dye amounts a_k, a_k += noise, back to logH (curves are monotonic)."""
    sig_px = prm["rms"] / 1000 * np.sqrt(np.pi * (prm["aperture_um"] / 2) ** 2) / um_per_px  # Selwyn: sigma*sqrt(area) const
    sig_unit = sig_px / np.sqrt((v_k ** 2).sum()) * prm["strength"]  # per layer at unit dye amount
    rng = np.random.default_rng(seed); out = np.empty_like(logH); c_px = prm["cloud_um"] / um_per_px
    rho = prm.get("layer_corr", 0.0); shared = rng.standard_normal(logH.shape[:2], dtype=np.float32) if rho > 0 else 0
    cv = fastops.curves(stock); logH = np.asarray(logH, np.float32)
    for k in range(3):
        a = cv.amount(logH[..., k], k)
        n = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(a.shape, dtype=np.float32)  # partly co-located clouds
        if c_px >= 0.5:  # clouds resolved: correlate and renormalise to unit variance
            n = gaussian_filter(n, c_px) * np.sqrt(4 * np.pi) * c_px
        e = prm.get("density_exp", 0.5); a0 = np.clip(a, 0, None)
        a2 = a + np.float32(sig_unit) * (a0 if e == 1.0 else a0 ** e) * n  # exp 1.0: scans show grain ~ density
        out[..., k] = cv.logH(a2, k)  # the table clamps to the curve's range
    return out.astype(np.float32), dict(sigma_density_per_px=round(float(sig_px), 4), sigma_unit_dye=round(float(sig_unit), 4), seed=seed)
