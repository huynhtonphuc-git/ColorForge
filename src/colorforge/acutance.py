"""Film acutance in log-exposure space (blueprint v2 D2 + D3), sized in um on the film frame.

Linearised Nelson chemical-spread model fitted to the stock's datasheet MTF:
  MTF(v) = MTF_opt(v) * [1 + A (1 - B(v))]
  MTF_opt(v) = 1 / (1 + (2 pi K v)^2)                              optical scatter, exponential line spread, K um
  B(v) = w exp(-2 pi^2 s_dev^2 v^2) + (1-w) exp(-2 pi^2 s_br^2 v^2)  chemical spread: developer + bromide, um
Applied as a radial filter on log10 H per layer (datasheet MTFs are measured in effective-exposure terms), so a
small-signal sine of log H is transferred exactly with the datasheet response.
Assumptions: same filter for all layers (datasheet gives one curve); phone MTF taken as 1 (not yet divided out);
the frame's long side = 36 mm.
"""
import numpy as np
from scipy import fft

# KR 64 (Kodak E-88 2009 p4) fit with A fixed at 1.0: max |err| 0.030; K vs A trade off (see HANDOFF) -> assumption
KODACHROME_64 = dict(K=7.54, A=1.00, s_dev=11.2, s_br=48.2, w=0.68,
                     provenance="assumption: fitted to Kodak E-88 (2009) KR 64 MTF, max |err| 0.030; K/A split not unique")


def mtf(p, v_cmm):
    f = np.asarray(v_cmm) / 1000.0  # cycles per um
    B = p["w"] * np.exp(-2 * np.pi ** 2 * p["s_dev"] ** 2 * f ** 2) + (1 - p["w"]) * np.exp(-2 * np.pi ** 2 * p["s_br"] ** 2 * f ** 2)
    return (1 + p["A"] * (1 - B)) / (1 + (2 * np.pi * p["K"] * f) ** 2)


def um_per_px(shape, frame_long_mm=36.0):
    return frame_long_mm * 1000.0 / max(shape[:2])


def apply(logH, p, frame_long_mm=36.0, workers=-1):
    """logH (H, W, 3) float32 -> filtered logH. Reflect-padded FFT so edges do not wrap."""
    H, W, _ = logH.shape; pitch = um_per_px(logH.shape, frame_long_mm)
    pad = int(np.ceil(3 * p["s_br"] / pitch))  # widest kernel: bromide spread
    x = np.pad(logH, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    fy = fft.fftfreq(x.shape[0], d=pitch / 1000.0)[:, None]  # cycles per mm
    fx = fft.rfftfreq(x.shape[1], d=pitch / 1000.0)[None, :]
    T = mtf(p, np.sqrt(fy ** 2 + fx ** 2)).astype(np.float32)
    out = np.empty_like(logH)
    for c in range(3):
        out[..., c] = fft.irfft2(fft.rfft2(x[..., c], workers=workers) * T, s=x.shape[:2], workers=workers)[pad:pad + H, pad:pad + W]
    return out
