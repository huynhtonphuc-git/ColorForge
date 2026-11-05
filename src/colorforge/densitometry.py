"""Status A densitometry (ISO 5-3) on the ColorForge 380-780 nm / 5 nm grid.

Status A log spectral products (R, G, B) as tabulated in spectral_film_lut/densiometry.py (MIT), which
reproduces ISO 5-3. Provenance: datasheet (standard). Integral density of a transmission spectrum T:
    D_c = -log10( sum T * 10**S_c / sum 10**S_c )
"""
import numpy as np

WL = np.arange(380.0, 781.0, 5.0)

_STATUS_A = [
    {599: 2.298, 600: 2.568, 610: 4.638, 620: 5.000, 630: 4.871, 640: 4.604, 650: 4.286, 660: 3.900, 670: 3.551,
     680: 3.165, 690: 2.776, 700: 2.383, 710: 1.970, 720: 1.551, 730: 1.141, 740: 0.741, 750: 0.341, 751: 0.301},
    {499: 1.430, 500: 1.650, 510: 3.822, 520: 4.782, 530: 5.000, 540: 4.906, 550: 4.644, 560: 4.221, 570: 3.609,
     580: 2.766, 590: 1.579, 591: 1.409},
    {419: 3.222, 420: 3.602, 430: 4.819, 440: 5.000, 450: 4.912, 460: 4.620, 470: 4.040, 480: 2.989, 490: 1.566,
     500: 0.165, 501: 0.015},
]


def _weights(table):
    w = np.zeros((len(WL), 3))
    for c, t in enumerate(table):
        x = np.array(sorted(t)); y = np.array([t[k] for k in x])
        w[:, c] = np.where((WL >= x[0]) & (WL <= x[-1]), 10 ** np.interp(WL, x, y), 0.0)
    return w / w.sum(0)


STATUS_A = _weights(_STATUS_A)  # (81, 3) normalised linear weights


def fill_edges(a):
    """Replace NaN at spectrum ends by the nearest valid value (assumption: flat beyond the digitised range)."""
    a = np.array(a, float); a = a[:, None] if a.ndim == 1 else a
    for c in range(a.shape[1]):
        m = ~np.isnan(a[:, c]); i = np.flatnonzero(m)
        a[:, c] = np.interp(np.arange(len(a)), i, a[i, c])
    return a.squeeze()


def status_a(spectral_density):
    """spectral_density (..., 81) -> Status A R, G, B integral density (..., 3)."""
    T = 10.0 ** -np.asarray(spectral_density, float)
    return -np.log10(np.clip(T @ STATUS_A, 1e-12, None))
