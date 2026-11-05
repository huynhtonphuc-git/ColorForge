"""End of Input Adapt: repair and enhance the capture before any film sees it. The film then "photographs" this result.

Order on the scene-linear image: phone blur removal -> AI micro-contrast (enhance.py, off by default) -> tonal enhancement (off by default).
Both enhancements are the same gain on all three channels, so colour ratios of the scene stay as they were. They run only on low-bit input (route.plan).
Tonal enhancement values are `assumption`, tuned by measurement on scans.
"""
import cv2
import numpy as np
from scipy import fft

import acutance2

TONE_K = 0.3; TONE_RADIUS_UM = 400.0; TONE_EPS = 0.01
LUMA = np.array([0.2627, 0.6780, 0.0593], np.float32)  # BT.2020


def remove_phone_blur(scene, pitch_um, phone):
    """Wiener inverse of the phone's measured MTF, per channel, in linear light. Keeps the photo's own mean."""
    H, W, _ = scene.shape; pad = 32; shape = (fft.next_fast_len(H + 2 * pad, real=True), fft.next_fast_len(W + 2 * pad, real=True)); f = acutance2._radial(shape, pitch_um).astype(np.float32); out = np.empty_like(scene)
    for k in range(3):
        out[..., k] = acutance2._filt(np.ascontiguousarray(scene[..., k]), acutance2.phone_wiener(f, pitch_um, k, phone).astype(np.float32), pad, shape)
    return out


def apply_detail(scene, detail_log10):
    """A brightness detail field, in log10, as a gain on all channels."""
    return scene * (10 ** detail_log10)[..., None]


def tone_detail(scene, um_per_px, k=TONE_K):
    """Local contrast at a few hundred um, as a log10 gain field. Guided filter on log brightness at 1/4 size: big edges stay out of the base, so no halos."""
    L = np.log10(np.maximum(np.asarray(scene, np.float32) @ LUMA, 1e-5)); H, W = L.shape; s = 4; g = cv2.resize(L, (max(1, W // s), max(1, H // s)), interpolation=cv2.INTER_AREA)
    r = max(1, int(round(TONE_RADIUS_UM / um_per_px / s))); box = lambda a: cv2.blur(a, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    m = box(g); v = box(g * g) - m * m; a = v / (v + TONE_EPS); b = m - a * m
    A, B = (cv2.resize(box(x), (W, H), interpolation=cv2.INTER_LINEAR) for x in (a, b))
    return np.float32(k) * (L - (A * L + B))
