"""Display encodings and file writers: sRGB and Display P3 (JPEG 8-bit + TIFF 16-bit) with embedded ICC profiles.

Input is display-referred linear BT.2020 (D65, white = 1). Out-of-gamut colours are clipped per channel for now
(TODO: chroma compression in a UCS, see research report 'Kodachrome punch and acutance').
ICC: minimal v2 matrix/TRC display profiles generated here (primaries Bradford-adapted to D50 as ICC requires).
"""
import io, struct
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

XYZ_TO_BT2020 = np.array([[1.7166512, -0.3556708, -0.2533663], [-0.6666844, 1.6164812, 0.0157685], [0.0176399, -0.0427706, 0.9421031]])
SPACES = {"srgb": [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], "p3": [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)]}
D65_XY = (0.3127, 0.3290)
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def rgb_to_xyz(prim, white=D65_XY):
    P = np.array([[x / y, 1, (1 - x - y) / y] for x, y in prim]).T
    W = np.array([white[0] / white[1], 1, (1 - white[0] - white[1]) / white[1]])
    return P * np.linalg.solve(P, W)


def from_bt2020(rgb2020, space):
    M = np.linalg.inv(rgb_to_xyz(SPACES[space])) @ np.linalg.inv(XYZ_TO_BT2020)
    return rgb2020 @ M.T.astype(np.float32)


def srgb_oetf(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def icc_profile(space, desc):
    return icc_profile_prim(SPACES[space], desc)


def icc_profile_prim(primaries, desc):
    """Minimal ICC v2 display profile: sRGB transfer curve (1024-entry curv) + Bradford-D50 primaries."""
    M = rgb_to_xyz(primaries); W65 = M @ [1, 1, 1]; W50 = np.array([0.9642, 1.0, 0.8249])
    A = np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ W50) / (BRADFORD @ W65)) @ BRADFORD; M50 = A @ M
    s15 = lambda v: struct.pack(">i", int(round(v * 65536)))
    xyz = lambda v: b"XYZ " + b"\0" * 4 + b"".join(s15(c) for c in v)
    curve = (srgb_oetf_inverse(np.linspace(0, 1, 1024)) * 65535).round().astype(">u2").tobytes()
    curv = b"curv" + b"\0" * 4 + struct.pack(">I", 1024) + curve
    text = lambda s: b"desc" + b"\0" * 4 + struct.pack(">I", len(s) + 1) + s.encode() + b"\0" + b"\0" * 12 + b"\0" * 67
    tags = [(b"desc", text(desc)), (b"wtpt", xyz(W50)), (b"rXYZ", xyz(M50[:, 0])), (b"gXYZ", xyz(M50[:, 1])), (b"bXYZ", xyz(M50[:, 2])),
            (b"rTRC", curv), (b"gTRC", curv), (b"bTRC", curv), (b"cprt", b"text" + b"\0" * 4 + b"Public domain (ColorForge)\0")]
    off = 128 + 4 + 12 * len(tags); table = b""; data = b""
    for sig, body in tags:
        while (off + len(data)) % 4:
            data += b"\0"
        table += sig + struct.pack(">II", off + len(data), len(body)); data += body
    size = off + len(data)
    hdr = struct.pack(">I4s4s4s4s4s12s4s4sI4s4sQI12s4s16s28s", size, b"none", b"\x02\x10\x00\x00", b"mntr", b"RGB ", b"XYZ ", b"\0" * 12, b"acsp",
                      b"MSFT", 0, b"none", b"none", 0, 0, s15(0.9642) + s15(1.0) + s15(0.8249), b"none", b"\0" * 16, b"\0" * 28)
    return hdr + struct.pack(">I", len(tags)) + table + data


def srgb_oetf_inverse(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def compress_gamut(rgb, threshold=0.80, limit=1.20, power=1.2):
    """Smooth gamut compression in the target RGB (ACES Reference Gamut Compression form, per-channel distance from the
    achromatic axis): distances below `threshold` untouched, distances up to `limit` (slightly negative channels) mapped
    onto the gamut boundary. Hue is approximately kept, lightness kept; replaces hard clipping. Only touches pixels that need it."""
    rgb = np.asarray(rgb, np.float32); ach = rgb.max(-1, keepdims=True); safe = np.maximum(np.abs(ach), 1e-6); d = (ach - rgb) / safe
    m = d >= threshold
    if not m.any():
        return rgb.copy()
    s = gamut_scale(threshold, limit, power); x = (d[m] - threshold) / s
    d[m] = threshold + s * x / (1 + x ** power) ** (1 / power)
    return ach - d * safe


def gamut_scale(threshold=0.80, limit=1.20, power=1.2):
    return (limit - threshold) / (((1 - threshold) / (limit - threshold)) ** -power - 1) ** (1 / power)


def write_hdr_jpeg(sdr_lin_p3, hdr_lin_p3, path, quality=95):
    """ISO 21496-1 / Ultra HDR JPEG via libultrahdr: SDR base (Display P3, sRGB transfer, 8-bit) + gain map computed
    from the HDR rendition (linear Display P3, 1.0 = 203 cd/m2 SDR reference white)."""
    import imagecodecs as ic
    H, W, _ = sdr_lin_p3.shape
    sdr = np.empty((H, W, 4), np.uint8); sdr[..., :3] = (srgb_oetf(sdr_lin_p3) * 255 + 0.5).astype(np.uint8); sdr[..., 3] = 255
    hdr = np.empty((H, W, 4), np.float16); hdr[..., :3] = np.clip(hdr_lin_p3, 0, 64); hdr[..., 3] = 1
    U = ic.ULTRAHDR
    data = ic.ultrahdr_encode(hdr, level=quality, sdr=sdr, gamut=U.CG.DISPLAY_P3, transfer=U.CT.LINEAR, crange=U.CR.FULL_RANGE,
                              usage=U.USAGE.QUALITY, codec=U.CODEC.JPEG)
    path.write_bytes(data)
    return len(data)


_v = np.arange(65536) / 65535
_ENC8 = np.round(np.where(_v <= 0.0031308, 12.92 * _v, 1.055 * _v ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)  # linear 16-bit -> sRGB 8-bit


def encode_jpeg(lin, quality=90, icc=None):
    """Linear [0,1] RGB -> JPEG bytes with sRGB transfer; icc=None means sRGB, otherwise the profile of the primaries."""
    buf = io.BytesIO(); kw = dict(icc_profile=icc) if icc else {}
    Image.fromarray(_ENC8[(np.clip(lin, 0, 1) * 65535 + 0.5).astype(np.uint16)]).save(buf, "JPEG", quality=quality, subsampling=0 if quality >= 92 else 2, **kw)
    return buf.getvalue()


def write_jxl(lin_p3, path):
    """16-bit lossless JPEG XL, Display P3 primaries, sRGB transfer. Lossless because lossy modes smear film grain."""
    import imagecodecs as ic
    u16 = (srgb_oetf(lin_p3) * 65535 + 0.5).astype(np.uint16)
    Path(path).write_bytes(ic.jpegxl_encode(u16, lossless=True, effort=3, primaries=ic.JPEGXL.PRIMARIES.P3, transfer=ic.JPEGXL.TRANSFER_FUNCTION.SRGB, bitspersample=16))


def write_linear_dng(lin_p3, path):
    """16-bit LinearRaw DNG, scene-style linear Display P3 (the film's look is baked in; values may exceed SDR white).
    Peak is normalised to the white level and BaselineExposure restores it. Identity tone curve, so raw converters do not add their own."""
    peak = max(1.0, float(np.percentile(lin_p3.max(-1), 99.99)))
    u16 = (np.clip(lin_p3 / peak, 0, 1) * 65535 + 0.5).astype(np.uint16)
    M = rgb_to_xyz(SPACES["p3"]); W65 = M @ [1, 1, 1]; W50 = np.array([0.96422, 1.0, 0.82521])
    FM = np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ W50) / (BRADFORD @ W65)) @ BRADFORD @ M  # white-balanced camera RGB -> XYZ D50
    srat = lambda m: tuple(int(v) for x in np.ravel(m) for v in (round(float(x) * 10000), 10000))
    tags = [(50706, "B", 4, (1, 4, 0, 0), True), (50707, "B", 4, (1, 1, 0, 0), True), (50708, "s", 0, "ColorForge FilmForge Kodachrome 64", True), (271, "s", 0, "ColorForge", True), (272, "s", 0, "FilmForge", True),
            (50717, "I", 1, (65535,), True), (50721, "2i", 9, srat(np.linalg.inv(M)), True), (50964, "2i", 9, srat(FM), True), (50778, "H", 1, (21,), True),
            (50728, "2I", 3, (1, 1, 1, 1, 1, 1), True), (50730, "2i", 1, srat([np.log2(peak)]), True), (50940, "f", 4, (0.0, 0.0, 1.0, 1.0), True)]
    tifffile.imwrite(path, u16, photometric=34892, planarconfig="contig", extratags=tags, metadata=None)


def write_outputs(rgb2020, stem, out_dir, spaces=("srgb", "p3")):
    files = {}
    for sp in spaces:
        enc = srgb_oetf(from_bt2020(rgb2020, sp)); icc = icc_profile(sp, "ColorForge " + ("sRGB" if sp == "srgb" else "Display P3"))
        j = out_dir / f"{stem}_{sp}.jpg"; t = out_dir / f"{stem}_{sp}_16bit.tif"
        Image.fromarray((enc * 255 + 0.5).astype(np.uint8)).save(j, quality=95, icc_profile=icc)
        tifffile.imwrite(t, (enc * 65535 + 0.5).astype(np.uint16), compression="zlib", extratags=[(34675, 7, len(icc), icc, True)])
        files[sp] = [j.name, t.name]
    return files
