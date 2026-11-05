"""ColorForge DNG -> CF-Scene converter (v0.1, milestone 1).

CF-Scene = scene-linear BT.2020 (D65) float32 + a capture report.
Pipeline: decode tiles (own lossless-JPEG decoder) -> black/white level -> OpcodeList2 GainMap
-> demosaic -> DNG colour (AsShotNeutral, CameraCalibration, interpolated ForwardMatrix) -> XYZ D50
-> Bradford D65 -> BT.2020 -> DefaultCrop -> Orientation.
Only numpy, scipy, numba, tifffile, opencv. Vendor look tables and tone curves are deliberately ignored.
"""
import json, struct, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

import fastops

ILLUM_CCT = {17: 2856.0, 18: 4874.0, 19: 6774.0, 20: 5503.0, 21: 6504.0, 22: 7504.0, 23: 4000.0, 24: 3400.0}
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
XYZ_D50 = np.array([0.96422, 1.0, 0.82521]); XYZ_D65 = np.array([0.95047, 1.0, 1.08883])
XYZ_TO_BT2020 = np.array([[1.7166512, -0.3556708, -0.2533663], [-0.6666844, 1.6164812, 0.0157685], [0.0176399, -0.0427706, 0.9421031]])
BT2020_TO_SRGB = np.array([[1.6604910, -0.5876411, -0.0728499], [-0.1245505, 1.1328999, -0.0083494], [-0.0181508, -0.1005789, 1.1187297]])


def _rat(v):
    v = list(v); return np.array([v[i] / v[i + 1] for i in range(0, len(v), 2)], float)


def _m3(v):
    return _rat(v).reshape(3, 3) if len(v) == 18 else np.asarray(v, float).reshape(3, 3)


def _decode_tile(args):
    data, th, tw = args
    out, info = fastops.lj92_decode(data)
    return out[:th, :tw]


def read_dng(path, workers=8):
    t = tifffile.TiffFile(path); p = t.pages[0]; tg = {x.name: x.value for x in p.tags.values()}
    assert p.photometric == 32803, "only mosaic (CFA) DNG supported so far"
    H, W = p.imagelength, p.imagewidth; th, tw = tg["TileLength"], tg["TileWidth"]
    fh = t.filehandle; jobs = []
    for o, c in zip(tg["TileOffsets"], tg["TileByteCounts"]):
        fh.seek(o); jobs.append((fh.read(c), th, tw))
    preview = None
    if p.pages:  # embedded phone preview: baseline JPEG strip
        sp = p.pages[0]; so = sp.tags["StripOffsets"].value; sc = sp.tags["StripByteCounts"].value
        so = so[0] if isinstance(so, (tuple, list)) else so; sc = sc[0] if isinstance(sc, (tuple, list)) else sc
        fh.seek(so); buf = np.frombuffer(fh.read(sc), np.uint8)
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        preview = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB) if dec is not None else None
    t.close()
    with ThreadPoolExecutor(workers) as ex:  # the compiled decoder releases the GIL
        tiles = list(ex.map(_decode_tile, jobs))
    nx = -(-W // tw); cfa = np.zeros((-(-H // th) * th, nx * tw), np.uint16)
    for k, tile in enumerate(tiles):
        y, x = divmod(k, nx); cfa[y * th:(y + 1) * th, x * tw:(x + 1) * tw] = tile
    return cfa[:H, :W], tg, preview


def parse_opcodes(b):
    b = bytes(b); n = struct.unpack(">I", b[:4])[0]; o = 4; ops = []
    for _ in range(n):
        oid, ver, flags, size = struct.unpack(">IIII", b[o:o + 16]); pay = b[o + 16:o + 16 + size]
        if oid == 9:
            top, left, bot, right, plane, planes, rp, cp, mv, mh = struct.unpack(">10I", pay[:40])
            sv, sh, ov, oh = struct.unpack(">4d", pay[40:72]); mp = struct.unpack(">I", pay[72:76])[0]
            g = np.frombuffer(pay[76:76 + 4 * mv * mh * mp], ">f4").astype(np.float64).reshape(mv, mh, mp)
            ops.append(dict(id=9, area=(top, left, bot, right), pitch=(rp, cp), spacing=(sv, sh), origin=(ov, oh), gain=g[..., 0]))
        elif oid == 1:
            pl = struct.unpack(">I", pay[:4])[0]; v = struct.unpack(">%dd" % (pl * 6 + 2), pay[4:4 + 8 * (pl * 6 + 2)])
            ops.append(dict(id=1, coeffs=v[:-2], center=v[-2:]))
        else:
            ops.append(dict(id=oid, skipped=True, optional=bool(flags & 1)))
        o += 16 + size
    return ops


def apply_gainmaps(img, ops, H, W):
    """img: normalised CFA (float). GainMap coordinates are relative to the active area."""
    for op in ops:
        if op["id"] != 9:
            continue
        top, left, bot, right = op["area"]; rp, cp = op["pitch"]
        ys = np.arange(top, bot, rp); xs = np.arange(left, right, cp)
        mv, mh = op["gain"].shape
        fy = ((ys + 0.5) / H - op["origin"][0]) / op["spacing"][0]
        fx = ((xs + 0.5) / W - op["origin"][1]) / op["spacing"][1]
        YY, XX = np.meshgrid(np.clip(fy, 0, mv - 1), np.clip(fx, 0, mh - 1), indexing="ij")
        g = map_coordinates(op["gain"], [YY, XX], order=1, mode="nearest")
        img[top:bot:rp, left:right:cp] *= g
    return img


def _cct_from_xy(x, y):  # McCamy
    n = (x - 0.3320) / (0.1858 - y); return 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33


def _interp(m1, m2, t1, t2, cct):
    w = np.clip((1 / cct - 1 / t2) / (1 / t1 - 1 / t2), 0, 1); return w * m1 + (1 - w) * m2


def camera_to_xyz_d50(tg):
    t1 = ILLUM_CCT.get(tg.get("CalibrationIlluminant1", 21), 6504.0); t2 = ILLUM_CCT.get(tg.get("CalibrationIlluminant2", 17), 2856.0)
    cm1, cm2 = _m3(tg["ColorMatrix1"]), _m3(tg.get("ColorMatrix2", tg["ColorMatrix1"]))
    cc1 = _m3(tg["CameraCalibration1"]) if "CameraCalibration1" in tg else np.eye(3)
    cc2 = _m3(tg["CameraCalibration2"]) if "CameraCalibration2" in tg else cc1
    ab = np.diag(_rat(tg["AnalogBalance"])) if "AnalogBalance" in tg else np.eye(3)
    neutral = _rat(tg["AsShotNeutral"]); cct = 5000.0
    for _ in range(30):  # find the white point consistent with AsShotNeutral (DNG spec iteration)
        xyz_to_cam = ab @ _interp(cc1, cc2, t1, t2, cct) @ _interp(cm1, cm2, t1, t2, cct)
        xyz = np.linalg.solve(xyz_to_cam, neutral); x, y = xyz[:2] / xyz.sum()
        new = float(np.clip(_cct_from_xy(x, y), 2000, 12000))
        if abs(new - cct) < 0.5:
            break
        cct = new
    cc = _interp(cc1, cc2, t1, t2, cct)
    if "ForwardMatrix1" in tg:
        fm = _interp(_m3(tg["ForwardMatrix1"]), _m3(tg.get("ForwardMatrix2", tg["ForwardMatrix1"])), t1, t2, cct)
        ref_neutral = np.linalg.solve(ab @ cc, neutral)
        M = fm @ np.diag(1 / ref_neutral) @ np.linalg.inv(ab @ cc); path = "ForwardMatrix"
    else:  # ColorMatrix path with Bradford to D50
        cam_to_xyz = np.linalg.inv(ab @ cc @ _interp(cm1, cm2, t1, t2, cct)); M = cam_to_xyz; path = "ColorMatrix"
    return M, dict(cct_K=round(cct), white_xy=[round(x, 4), round(y, 4)], neutral=neutral.round(4).tolist(), matrix_path=path)


def bradford(src, dst):
    s = BRADFORD @ src; d = BRADFORD @ dst; return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


def demosaic(cfa_lin, pattern):
    """Edge-aware OpenCV demosaic on 16-bit; pattern is DNG CFAPattern (0=R,1=G,2=B) for the 2x2 block."""
    # OpenCV names Bayer codes by a shifted 2x2 window, so DNG "BGGR" maps to OpenCV "RG" (verified on Redmi sample).
    code = {(0, 1, 1, 2): cv2.COLOR_BayerBG2RGB_EA, (2, 1, 1, 0): cv2.COLOR_BayerRG2RGB_EA,
            (1, 0, 2, 1): cv2.COLOR_BayerGB2RGB_EA, (1, 2, 0, 1): cv2.COLOR_BayerGR2RGB_EA}[tuple(pattern)]
    scale = 60000.0
    u16 = np.clip(cfa_lin * scale, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(u16, code).astype(np.float32) / scale


def to_cf_scene(path, denoiser=None):
    """denoiser: optional callable (cfa01, pattern, S, O) -> (cfa01, change_map, info), applied before gain maps."""
    t0 = time.time()
    cfa, tg, preview = read_dng(path); H, W = cfa.shape
    rep = dict(file=Path(path).name, make=tg.get("Make"), model=tg.get("Model"), software=tg.get("Software"))
    pattern = list(bytes(tg["CFAPattern"])) if not isinstance(tg["CFAPattern"], tuple) else list(tg["CFAPattern"])
    bl = tg["BlackLevel"]; bl = np.asarray(bl, float) if np.ndim(bl) == 0 else (_rat(bl) if len(bl) % 2 == 0 and tg.get("BlackLevelRepeatDim") and any(v > 1 and i % 2 for i, v in enumerate(bl)) else np.asarray(bl, float))  # rational pairs vs plain ints
    black = np.asarray(bl, float).reshape(2, 2) if bl.size == 4 else np.full((2, 2), float(bl.ravel()[0]))
    white = float(tg["WhiteLevel"] if np.isscalar(tg["WhiteLevel"]) else tg["WhiteLevel"][0])
    raw = cfa.astype(np.float64)
    clip = (raw >= white - 1)
    lin = np.empty_like(raw)
    for dy in range(2):
        for dx in range(2):
            lin[dy::2, dx::2] = (raw[dy::2, dx::2] - black[dy, dx]) / (white - black[dy, dx])
    ops2 = parse_opcodes(tg["OpcodeList2"]) if "OpcodeList2" in tg else []
    ops3 = parse_opcodes(tg["OpcodeList3"]) if "OpcodeList3" in tg else []
    dn_info = None
    if denoiser is not None and "NoiseProfile" in tg:  # noise model is valid on the raw mosaic, before lens-shading gain
        npf = np.asarray(tg["NoiseProfile"], float); S, O = (npf[2], npf[3]) if len(npf) >= 6 else (npf[0], npf[1])
        td = time.time(); out, change, dn_info = denoiser(lin.astype(np.float32), pattern, S, O)
        lin = np.where(clip, lin, out).astype(np.float64)  # never touch clipped photosites
        dn_info = dict(dn_info, seconds=round(time.time() - td, 1), change_sigma_median=float(np.median(change)), change_sigma_p99=float(np.percentile(change, 99)))
    lin = apply_gainmaps(lin, ops2, H, W)
    rgb_cam = demosaic(lin, pattern)
    M, col = camera_to_xyz_d50(tg)
    M_total = XYZ_TO_BT2020 @ bradford(XYZ_D50, XYZ_D65) @ M
    img = rgb_cam.reshape(-1, 3) @ M_total.T.astype(np.float32)
    img = img.reshape(H, W, 3)
    if clip.any():  # clipped channels lose their ratio and turn pink/magenta: blend to neutral at the brightest channel (film would go clear there)
        m = cv2.GaussianBlur(cv2.dilate(clip.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.5)[..., None]
        img = (1 - m) * img + m * img.max(-1, keepdims=True)
    if "DefaultCropOrigin" in tg:
        cy, cx = int(tg["DefaultCropOrigin"][1]), int(tg["DefaultCropOrigin"][0])
        ch, cw = int(tg["DefaultCropSize"][1]), int(tg["DefaultCropSize"][0])
        img = img[cy:cy + ch, cx:cx + cw]; clipc = clip[cy:cy + ch, cx:cx + cw]
    else:
        clipc = clip
    orient = int(tg.get("Orientation", 1))
    rot = {6: 3, 8: 1, 3: 2}.get(orient, 0)  # np.rot90 k (counter-clockwise)
    img = np.rot90(img, rot).copy(); clipc = np.rot90(clipc, rot)
    # exposure: BaselineExposure applied; mid-grey placed later by the film stage
    be = float(_rat(tg["BaselineExposure"])[0]) if "BaselineExposure" in tg else 0.0
    img *= 2 ** be
    Y = img.reshape(-1, 3) @ np.array([0.2627, 0.6780, 0.0593], np.float32)
    np_ = tg.get("NoiseProfile")
    rep.update(
        cf_scene=dict(space="BT.2020 linear, D65", dtype="float32", shape=list(img.shape)),
        capture=dict(exposure_s=float(_rat(tg["ExposureTime"])[0]) if "ExposureTime" in tg else None,
                     fnumber=float(_rat(tg["FNumber"])[0]) if "FNumber" in tg else None,
                     iso=tg.get("ISOSpeedRatings"), focal_mm=float(_rat(tg["FocalLength"])[0]) if "FocalLength" in tg else None),
        raw=dict(kind="mosaic CFA", bits=int(np.ceil(np.log2(white + 1))), white=white, black=float(black.mean()),
                 cfa_pattern="".join("RGB"[c] for c in pattern), baseline_exposure=be,
                 noise_reduction_applied=float(_rat(tg["NoiseReductionApplied"])[0]) if "NoiseReductionApplied" in tg else None,
                 noise_profile=[round(float(v), 9) for v in np_] if np_ is not None else None),
        opcodes=dict(list2=[("GainMap max %.2f" % o["gain"].max()) if o["id"] == 9 else o for o in ops2],
                     list3=[("WarpRectilinear identity" if o["id"] == 1 and abs(o["coeffs"][0] - 1) < 1e-9 and not any(o["coeffs"][1:]) else ("WarpRectilinear (not applied)" if o["id"] == 1 else o)) for o in ops3]),
        colour=col, denoise=dn_info,
        stats=dict(clipped_pixels_pct=round(100 * float(clipc.mean()), 3),
                   Y_percentiles={k: round(float(v), 5) for k, v in zip(["p1", "p50", "p99", "p99.9"], np.percentile(Y, [1, 50, 99, 99.9]))},
                   suggested_ev_to_mid_grey=round(float(np.log2(0.18 / max(np.percentile(Y, 50), 1e-6))), 2),
                   negative_values_pct=round(100 * float((img < 0).any(-1).mean()), 3)),
        seconds=round(time.time() - t0, 1))
    return img, rep, preview, rot


def srgb_preview(img, ev, width=1600):
    h, w = img.shape[:2]; s = width / w
    small = cv2.resize(img, (width, int(h * s)), interpolation=cv2.INTER_AREA)
    x = np.clip(small.reshape(-1, 3) @ BT2020_TO_SRGB.T.astype(np.float32), 0, None) * 2 ** ev
    x = x / (1 + x / 4)  # gentle highlight roll-off for inspection only
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055)
    return (np.clip(x.reshape(small.shape), 0, 1) * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); reps = []
    for f in sys.argv[2:]:
        img, rep, prev, rot = to_cf_scene(f); stem = Path(f).stem
        ev = rep["stats"]["suggested_ev_to_mid_grey"] - 1.0  # place median a stop under mid-grey for a natural indoor look
        cv2.imwrite(str(out / f"{stem}_cfscene_preview.jpg"), cv2.cvtColor(srgb_preview(img, ev), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if prev is not None:
            pv = np.rot90(prev, rot) if prev.shape[0] < prev.shape[1] and img.shape[0] > img.shape[1] else prev
            cv2.imwrite(str(out / f"{stem}_phone_preview.jpg"), cv2.cvtColor(np.ascontiguousarray(pv), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        np.save(out / f"{stem}_cfscene_half.npy", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA).astype(np.float16))
        reps.append(rep); print(json.dumps(rep, indent=1, default=str))
    (out / "capture_report.json").write_text(json.dumps(reps, indent=1, default=str))
