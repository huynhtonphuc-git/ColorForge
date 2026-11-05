"""Dye clouds: per-layer image texture derived from counting, not noise added on top. Prototype, not yet in the app.

Each layer's dye amount a at a point is a sum of clouds of size q that land at random (compound Poisson):
    variance = q a    so sigma(a) = sigma_ref sqrt(a / a_ref)    skew = sigma(a) / a
Thin areas hold few clouds, so their texture is spiky; dense areas hold many, so it is close to Gaussian. Given the datasheet RMS, the skew has no free parameter.
sigma_ref is set so the visual sum reads the datasheet RMS through its 48 um aperture at visual density 1.0 (Kodak E-55). Layers are independent: each has its own clouds.
Provenance: RMS `datasheet`. Cloud size `assumption`: no Kodachrome value is published; one reference gives 10-25 um diameter for colour film generally. Independence of the layers `assumption`.
"""
import cv2
import numpy as np

import fastops

APERTURE_UM = 48.0


def _field(shape, sig_px, rng):
    n = rng.standard_normal(shape, dtype=np.float32); return cv2.GaussianBlur(n, (0, 0), sig_px, borderType=cv2.BORDER_REFLECT) if sig_px >= 0.3 else n


def _aperture_gain(sig_px, um_per_px):
    """Multiplier that makes a unit-std field read std 1 through the datasheet's circular aperture, measured instead of assumed (Selwyn's law fails near the cloud size)."""
    g = _field((1024, 1024), sig_px, np.random.default_rng(1)); g = g / g.std(); r = APERTURE_UM / 2 / um_per_px; R = int(np.ceil(r)) + 1; yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    sub = np.linspace(-0.45, 0.45, 5); disc = sum(((xx + a) ** 2 + (yy + b) ** 2 <= r * r).astype(np.float32) for a in sub for b in sub); disc /= disc.sum()
    return 1.0 / float(cv2.filter2D(g, -1, disc, borderType=cv2.BORDER_REFLECT)[R:-R, R:-R].std())


def apply(stock, logH, um_per_px, v_k, cloud_um=5.0, strength=1.0, seed=0, skew=True):
    """logH (H, W, 3) -> logH with clouds. cloud_um: Gaussian sigma of a cloud on the film."""
    logH = np.asarray(logH, np.float32); rng = np.random.default_rng(seed); cv = fastops.curves(stock); sig_px = cloud_um / um_per_px; gain = _aperture_gain(sig_px, um_per_px)
    a_ref = np.array([float(np.interp(1.0, stock.D[::-1, 1], stock.amounts[::-1, c])) for c in range(3)]); sig_ref = stock.rms / 1000 / float(np.sqrt(((v_k * np.sqrt(a_ref)) ** 2).sum() / a_ref.mean())) * gain * strength / np.sqrt(a_ref.mean()) * np.sqrt(a_ref)  # layer sigma at its own a_ref
    out = np.empty_like(logH)
    for k in range(3):
        a = np.maximum(cv.amount(logH[..., k], k), 1e-4); s = np.float32(sig_ref[k]) * np.sqrt(a / np.float32(a_ref[k])); n = _field(a.shape, sig_px, rng); n /= n.std()
        if skew:  # standardised lognormal with skewness about s/a: for small skew, shape w = skew / 3
            w = np.minimum(s / a / 3.0, 0.6); w2 = w * w; n = (np.exp(w * n) - np.exp(w2 / 2)) / np.maximum(np.sqrt((np.exp(w2) - 1) * np.exp(w2)), 1e-4)
            n = np.where(w < 1e-3, 0.0, n) + np.where(w < 1e-3, 1.0, 0.0) * _small(n)
        out[..., k] = cv.logH(np.maximum(a + s * n, 0.0), k)
    return out


def _small(n):
    return np.zeros_like(n)
