"""Camera raw -> CF-Scene, the same way as the phone DNG path: raw mosaic -> AI denoise -> demosaic -> lens correction -> colour, all in float.

LibRaw (rawpy) only unpacks the file and supplies its metadata. Its finished image is not used for Bayer sensors, because that is clipped to 0-1 and has no mosaic left to denoise.
X-Trans and other non-Bayer sensors return None here and take the LibRaw image in ingest.py (no AI denoise: the model needs a 2x2 Bayer mosaic).

Colour: XYZ(D65) -> camera matrix from LibRaw (the DNG ColorMatrix for this camera), scaled so camera white = D65, then camera white balance. Out-of-range values stay.
Noise: variance = a x + b on the mosaic, estimated from the photo itself. Against makers' DNG NoiseProfile tags on single-frame files: sigma x1.01-1.20; it reads high on very clean files
(about x2 in a synthetic test at a = 3e-6). darktable's published table is not used: on the cameras compared its values sit 13-22x from this scale, and a guessed factor would be invented data.
Lens: Lensfun (database CC BY-SA 3.0, library LGPL-3): vignetting, chromatic aberration, distortion. Output keeps the input's size.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import rawpy

from cf_dng import XYZ_D65, XYZ_TO_BT2020, demosaic


def exif_info(path):
    """Make, model, lens, focal length, aperture, ISO. Missing values come back as None."""
    import exifread
    out = dict(make=None, model=None, lens_make=None, lens=None, focal_mm=None, focal_35mm=None, fnumber=None, iso=None, exposure_s=None)
    try:
        with open(path, "rb") as f:
            t = exifread.process_file(f, details=False)
    except Exception:
        return out
    txt = lambda *k: next((str(t[x]).strip() for x in k if x in t and str(t[x]).strip() not in ("", "0")), None)

    def num(*k):
        for x in k:
            if x in t:
                try:
                    v = t[x].values[0]; v = float(v.num) / float(v.den) if hasattr(v, "num") else float(v)
                    if v > 0:
                        return v
                except Exception:
                    pass
    out.update(make=txt("Image Make"), model=txt("Image Model"), lens_make=txt("EXIF LensMake"), lens=txt("EXIF LensModel", "MakerNote LensType"), focal_mm=num("EXIF FocalLength"), focal_35mm=num("EXIF FocalLengthIn35mmFilm"), fnumber=num("EXIF FNumber"),
               iso=num("EXIF ISOSpeedRatings", "EXIF RecommendedExposureIndex", "EXIF StandardOutputSensitivity", "Image ISOSpeedRatings"), exposure_s=num("EXIF ExposureTime"))
    return out


# ---- noise
def estimate_noise(cfa01, pattern, B=8, pct=5.0):
    """Noise of this photo, on the green sites of the mosaic: var = a x + b. Neighbouring greens see the same scene with independent noise, and the quietest patches at each level leave out texture."""
    from scipy.stats import chi2
    idx = [i for i, c in enumerate(pattern) if c == 1]
    if len(idx) != 2:
        return None
    (y1, x1), (y2, x2) = [(i // 2, i % 2) for i in idx]; g1 = cfa01[y1::2, x1::2]; g2 = cfa01[y2::2, x2::2]; H, W = min(g1.shape[0], g2.shape[0]), min(g1.shape[1], g2.shape[1]); g1, g2 = g1[:H, :W], g2[:H, :W]
    d = g1 - g2; d = d - cv2.blur(d, (B * 2 + 1, B * 2 + 1)); grid = lambda a: cv2.blur(a, (B, B))[B // 2::B, B // 2::B]
    m = grid(0.5 * (g1 + g2)); dm = grid(d); v = (grid(d * d) - dm * dm) * B * B / (B * B - 1); expected = chi2.ppf(pct / 100, B * B - 1) / (B * B - 1)  # that percentile of the sample variance of pure noise
    edges = np.geomspace(max(1e-4, float(np.percentile(m, 1))), max(2e-3, float(np.percentile(m, 99))), 18); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (m >= lo) & (m < hi) & (m < 0.9)
        if sel.sum() >= 40:
            xs.append(float(np.median(m[sel]))); ys.append(float(np.percentile(v[sel], pct)) / 2 / expected)
    if len(xs) < 4:
        return None
    x = np.array(xs); y = np.array(ys); a, b = np.polyfit(x, y, 1, w=1 / np.maximum(y, 1e-12))
    return dict(a=float(max(a, 1e-7)), b=float(max(b, 1e-9)), points=len(xs))


# ---- lens
MAKER_FAMILY = {"om": ("olympus", "om system", "om digital"), "olympus": ("olympus", "om system", "om digital"), "nikon": ("nikon",), "sony": ("sony",), "canon": ("canon",), "fujifilm": ("fujifilm", "fujian"), "panasonic": ("panasonic", "leica", "lumix"), "pentax": ("pentax", "smc pentax", "hd pentax"), "ricoh": ("pentax", "ricoh")}


def _lens_by_name(db, make, name):
    """For a body Lensfun does not know yet: the same maker's lens with the same focal range and aperture, and only if exactly one fits."""
    nums = lambda t: re.findall(r"\d+(?:\.\d+)?", re.sub(r"\b(ii|iii)\b", "", t.lower())); want = nums(name)[:3]
    if len(want) < 2:
        return None
    fam = MAKER_FAMILY.get(make.lower().split()[0], (make.lower().split()[0],))
    hits = [l for l in db.lenses if any(f in (l.maker or "").lower() for f in fam) and [float(x) for x in nums(l.model)[:len(want)]] == [float(x) for x in want]]
    if len(hits) > 1:  # e.g. Sony E 35mm f/1.8 and FE 35mm f/1.8: the name's own words decide
        words = set(re.findall(r"[a-z]+", name.lower())) - {"f", "mm"}; hits = [l for l in hits if words <= set(re.findall(r"[a-z]+", l.model.lower()))]
    return hits[0] if len(hits) == 1 else None


def lens_modifier(info, W, H):
    """Lensfun correction for this photo, or (None, why). A wrong lens is worse than none, so a guess is never accepted: the file must name the lens, or the camera must have a fixed lens."""
    import lensfunpy
    if not (info.get("make") and info.get("model")):
        return None, "no camera name in the file"
    if not info.get("focal_mm"):
        return None, "no focal length in the file"
    db = lensfunpy.Database(); cams = db.find_cameras(info["make"], info["model"]); cam = cams[0] if cams else None; name = info.get("lens"); lens = None
    if cam is not None and name:
        found = db.find_lenses(cam, info.get("lens_make") or None, name); lens = found[0] if found else None
    elif cam is not None:  # no lens name: acceptable only when this camera can have just one lens
        found = db.find_lenses(cam); lens = found[0] if len(found) == 1 else None
    elif cam is None and name:
        lens = _lens_by_name(db, info["make"], name)
    if lens is None:
        return None, f"lens not identified: {name or 'not named in the file'}"
    crop = cam.crop_factor if cam is not None else (info["focal_35mm"] / info["focal_mm"] if info.get("focal_35mm") else lens.crop_factor)
    mod = lensfunpy.Modifier(lens, crop, W, H); mod.initialize(info["focal_mm"], info.get("fnumber") or lens.min_aperture or 5.6, 1000.0, pixel_format=np.float32, flags=lensfunpy.ModifyFlags.ALL)
    return (mod, f"{lens.maker} {lens.model}"), None


def correct_lens(rgb, info):
    """Vignetting, then per-channel distortion and chromatic aberration, in linear camera RGB. Returns image, report."""
    H, W = rgb.shape[:2]; got, why = lens_modifier(info, W, H)
    if got is None:
        return rgb, dict(applied=False, reason=why)
    mod, name = got; out = np.ascontiguousarray(rgb, np.float32).copy(); peak = float(out.max()); s = max(peak, 1e-6)
    out /= s; vig = bool(mod.apply_color_modification(out)); out *= s
    uv = mod.apply_subpixel_geometry_distortion()
    if uv is None:
        return out, dict(applied=vig, lens=name, vignetting=vig, geometry=False)
    uv = uv.astype(np.float32); moved = float(np.percentile(np.abs(uv[..., 1, :] - np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1)), 99.9))
    res = np.stack([cv2.remap(out[..., c], uv[..., c, 0], uv[..., c, 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT) for c in range(3)], -1)
    return res, dict(applied=True, lens=name, vignetting=vig, geometry=True, largest_shift_px=round(moved, 1))


# ---- the path
def camera_to_xyz(r):
    """Balanced camera RGB -> XYZ D65, plus the white-balance multipliers."""
    A = np.asarray(r.rgb_xyz_matrix, float)[:3, :3]
    if not np.abs(A).sum():
        return None, None
    n = A @ XYZ_D65; C = A / n[:, None]; M = np.linalg.inv(C)
    wb = np.asarray(r.camera_whitebalance, float)[:3]
    if not (wb > 0).all():
        wb = np.asarray(r.daylight_whitebalance, float)[:3]
    if not (wb > 0).all():
        wb = 1 / n
    return M, wb / wb.min()  # same scale as LibRaw: the weakest channel keeps its full range


def load(path, denoiser=None, lens=True):
    """Returns (img, report) for a Bayer raw, or None for other sensors. denoiser: callable (cfa01, pattern, S, O) -> (cfa01, change, info)."""
    with rawpy.imread(str(path)) as r:
        if r.num_colors not in (3, 4) or np.asarray(r.raw_pattern).shape != (2, 2) or "".join(c for c in r.color_desc.decode()) not in ("RGBG", "RGGB", "GRBG", "GBRG", "BGGR"):
            return None
        M, wb = camera_to_xyz(r); wb_source = "as shot" if (np.asarray(r.camera_whitebalance, float)[:3] > 0).all() else "not in the file: camera's daylight value"
        if M is None:
            return None
        raw = np.asarray(r.raw_image_visible, np.float32); cols = np.asarray(r.raw_colors_visible); black = np.asarray(r.black_level_per_channel, np.float32); white = float(r.white_level); flip = r.sizes.flip
        fix = lambda c: 1 if c == 3 else int(c)
        pattern = [fix(cols[0, 0]), fix(cols[0, 1]), fix(cols[1, 0]), fix(cols[1, 1])]
        lv = black[cols]; cfa = (raw - lv) / np.maximum(white - lv, 1); clip = raw >= white * 0.995; bits = int(np.ceil(np.log2(max(white, 2) + 1)))
    if tuple(pattern) not in ((0, 1, 1, 2), (2, 1, 1, 0), (1, 0, 2, 1), (1, 2, 0, 1)):
        return None
    H, W = cfa.shape; H -= H % 2; W -= W % 2; cfa = cfa[:H, :W]; clip = clip[:H, :W]; info = exif_info(path)
    rep = dict(file=Path(path).name, make=info["make"], model=info["model"], raw=dict(bits=bits, white=white, black=[float(x) for x in black]), capture=dict(iso=info["iso"], exposure_s=info["exposure_s"], fnumber=info["fnumber"], focal_mm=info["focal_mm"], lens=info["lens"]),
               colour=dict(wb=[round(float(x), 3) for x in wb], wb_source=wb_source, path="LibRaw camera matrix (XYZ D65)"))
    est = estimate_noise(cfa, pattern); rep["noise"] = est
    if denoiser is not None and est:
        out, change, dn = denoiser(cfa, pattern, est["a"], est["b"]); cfa = np.where(clip, cfa, out); rep["denoise"] = dict(dn, source="estimated from this photo", change_sigma_median=float(np.median(change)))
    rgb = demosaic(cfa, pattern) * wb.astype(np.float32); rep["lens_correction"] = dict(applied=False, reason="switched off")
    if lens:
        rgb, rep["lens_correction"] = correct_lens(rgb, info)
    img = (rgb.reshape(-1, 3) @ (XYZ_TO_BT2020 @ M).T.astype(np.float32)).reshape(rgb.shape)
    if clip.any():  # as in the DNG path: clipped channels lose their ratio, so blend to neutral at the brightest channel
        m = cv2.GaussianBlur(cv2.dilate(clip.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.5)[..., None]; img = (1 - m) * img + m * img.max(-1, keepdims=True)
    rep["clipped_pct"] = round(float(clip.mean() * 100), 3); rep["stats"] = dict(negative_pct=round(float((img < 0).any(-1).mean() * 100), 2), clipped_pixels_pct=rep["clipped_pct"])
    return np.rot90(img, {3: 2, 5: 1, 6: 3}.get(flip, 0)).copy(), rep
