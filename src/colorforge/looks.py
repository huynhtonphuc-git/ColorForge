"""Look engine: a grade after the film, kept apart from film data. Kodachrome's physics stays untouched; a look only changes the finished display image. Not film data: fitted to published pictures.

A look is a few bounded controls in CIELAB (D65): a tone curve, a chroma gain and hue turn for each of 7 hue families, and extra chroma in dark colours. Near-neutral colours are left alone, and the skin family has tight
limits, so skin is not pulled along with the colour changes. Parameters live in data/looks/<id>.json. They hold numbers only, no images.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data" / "looks"
HUES = np.array([30.0, 62.0, 95.0, 135.0, 215.0, 270.0, 340.0]); NAMES = ("red", "skin_orange", "yellow", "green", "cyan", "blue", "magenta"); SKIN = 1
TONE_L = np.array([0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0])
P3 = np.array([[0.48657, 0.26567, 0.19822], [0.22897, 0.69174, 0.07929], [0.0, 0.04511, 1.04394]], np.float32); P3I = np.linalg.inv(P3).astype(np.float32); WP = np.array([0.95047, 1.0, 1.08883], np.float32); E = 216 / 24389; K = 24389 / 27


def to_lab(xyz):
    t = xyz / WP; f = np.where(t > E, np.cbrt(np.maximum(t, 1e-9)), (K * t + 16) / 116); return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def from_lab(lab):
    fy = (lab[..., 0] + 16) / 116; f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1); return np.where(f ** 3 > E, f ** 3, (116 * f - 16) / K) * WP


def _circ(h, values):
    """Smooth over the hue circle, between the families."""
    x = np.r_[HUES[-1] - 360, HUES, HUES[0] + 360]; y = np.r_[values[-1], values, values[0]]; return np.interp(np.where(h < x[0], h + 360, h), x, y)


def grade_lab(lab, p, strength=1.0):
    """lab (..., 3) -> graded. p: dict(tone=5 offsets, gain=7, turn=7 degrees, dark=one value). strength 1 is the fitted look."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]; C = np.hypot(a, b); h = np.degrees(np.arctan2(b, a)) % 360; o = np.r_[0.0, np.asarray(p["tone"], float) * strength, 0.0]
    L2 = np.interp(L, TONE_L, np.maximum.accumulate(TONE_L + o)) + np.maximum(L - 100, 0) + np.minimum(L, 0)  # HDR values above white pass through
    w = np.clip(C / 10.0, 0, 1); w = w * w * (3 - 2 * w)  # greys stay
    g = 1 + (_circ(h, np.asarray(p["gain"], float)) - 1) * strength * w; g = g * (1 + float(p["dark"]) * strength * w * np.clip((60 - L) / 40, 0, 1)); t = np.radians(h + _circ(h, np.asarray(p["turn"], float)) * strength * w); C2 = C * g
    return np.stack([L2, C2 * np.cos(t), C2 * np.sin(t)], -1).astype(np.float32)


def apply(p3, look, strength=1.0):
    """Display-linear Display P3 -> same, graded. Out-of-gamut values are clipped by the caller's encoder."""
    sh = p3.shape; lab = to_lab(np.clip(p3, 0, None).reshape(-1, 3) @ P3.T); return (from_lab(grade_lab(lab, look, strength)) @ P3I.T).reshape(sh).astype(np.float32)


def load(look_id):
    return json.loads((OUT / f"{look_id}.json").read_text())
