"""Universal input: any supported file -> CF-Scene (scene-linear BT.2020 D65 float32) + capture report.

Route by content, not only by extension:
  mosaic DNG with lossless-JPEG tiles and a 2x2 Bayer pattern -> cf_dng (own decoder, GainMaps, NoiseProfile, PMRID denoise)
  other Bayer raw                                             -> camera_raw: LibRaw unpacks, then AI denoise, demosaic, lens correction, colour in float
  X-Trans and other non-Bayer raw                             -> LibRaw (rawpy) finished image: linear XYZ (D65), camera white balance
  DNG 1.7 LinearRaw with JPEG XL compression (stacked phone raws) -> own path: tifffile + imagecodecs, DNG colour maths
  JPEG with an Ultra HDR / ISO 21496-1 gain map               -> libultrahdr: linear HDR (1.0 = 203 cd/m2), used as scene-linear
  JPEG XL, JPEG, PNG, TIFF, WebP, AVIF                        -> decode, ICC-aware when possible, inverse transfer curve -> linear;
                                                                 report flags scene_estimated: the camera's tone curve stays baked in
report["source_class"] is one of raw-own, raw-libraw, raw-linear-dng, hdr-gainmap, encoded-sdr, encoded-float. report["phone_profile"] names a measured
phone MTF only when the file's Make/Model has one (so the Redmi blur correction never touches other cameras).
Not yet: AI dynamic-range extension for gain-map-less JPEG (allowed by CLAUDE.md, must add a mask and flag "AI-extended HDR").
"""
import io
from pathlib import Path

import imagecodecs as ic
import numpy as np
import rawpy
from PIL import Image, ImageCms

import output
import camera_raw
from cf_dng import XYZ_TO_BT2020, to_cf_scene

Image.MAX_IMAGE_PIXELS = None
RAW_EXT = {".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".rwl", ".pef", ".3fr", ".fff", ".iiq", ".srw", ".x3f", ".erf", ".mrw", ".kdc", ".dcr"}
BT2020_PRIM = [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)]


def phone_profile(make, model):
    text = f"{make} {model}".lower()
    return "REDMI_TURBO4PRO" if "turbo 4 pro" in text or "25053rt47c" in text else None


def _is_own_dng(path):
    import tifffile
    try:
        with tifffile.TiffFile(path) as t:
            p = t.pages[0]
            return p.photometric == 32803 and p.compression == 7 and "TileOffsets" in p.tags and len(bytes(p.tags["CFAPattern"].value)) == 4
    except Exception:
        return False


def _linear_dng(path):
    """DNG with already-demosaiced LinearRaw data (DNG 1.7 / JPEG XL compression, e.g. stacked phone raws): decode with tifffile +
    imagecodecs, apply black/white and the DNG colour path (same maths as cf_dng). ProfileGainTableMap and look tables are ignored."""
    import tifffile
    from cf_dng import _rat, bradford, camera_to_xyz_d50, XYZ_D50, XYZ_D65
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]; tg = {x.name: x.value for x in p.tags.values()}; bl_rational = "BlackLevel" in p.tags and p.tags["BlackLevel"].dtype in (5, 10)
        if p.photometric != 34892:
            raise ValueError("not LinearRaw")
        a = p.asarray().astype(np.float32)
    bl = tg.get("BlackLevel", (0,)); bl = _rat(bl) if bl_rational else np.asarray(bl, float)
    white = np.asarray(tg.get("WhiteLevel", 65535), float)
    lin = (a - np.float32(np.mean(bl))) / np.float32(np.mean(white) - np.mean(bl))
    M, col = camera_to_xyz_d50(tg)
    img = (lin.reshape(-1, 3) @ (XYZ_TO_BT2020 @ bradford(XYZ_D50, XYZ_D65) @ M).T.astype(np.float32)).reshape(lin.shape)
    if "DefaultCropOrigin" in tg:
        cx, cy = int(tg["DefaultCropOrigin"][0]), int(tg["DefaultCropOrigin"][1]); cw, ch = int(tg["DefaultCropSize"][0]), int(tg["DefaultCropSize"][1])
        img = img[cy:cy + ch, cx:cx + cw]
    rot = {6: 3, 8: 1, 3: 2}.get(int(tg.get("Orientation", 1)), 0)
    img = np.rot90(img, rot).copy()
    be = float(_rat(tg["BaselineExposure"])[0]) if "BaselineExposure" in tg else 0.0
    return img * np.float32(2 ** be), dict(make=str(tg.get("Make", "")), model=str(tg.get("Model", "")), colour=col, baseline_exposure=be,
                                           compression=int(p.compression), dng_version=list(tg.get("DNGVersion", b"")), note="LinearRaw DNG: stacked/demosaiced by the phone; ignored gain-table and look data")


def _libraw(path):
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(output_color=rawpy.ColorSpace.XYZ, gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True,
                            highlight_mode=rawpy.HighlightMode.Clip, user_flip=None)
        xyz = rgb.astype(np.float32) / 65535.0
        sizes = r.sizes; ncol = r.num_colors; pat = r.raw_pattern; wl = int(r.white_level)
    img = (xyz.reshape(-1, 3) @ XYZ_TO_BT2020.T.astype(np.float32)).reshape(xyz.shape)
    return img, dict(bits=int(np.ceil(np.log2(max(wl, 2) + 1))), raw_width=sizes.raw_width, raw_height=sizes.raw_height, num_colors=ncol, pattern_shape=None if pat is None else list(pat.shape))


def _exif(path):
    make = model = ""
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            tg = t.pages[0].tags
            make = str(tg["Make"].value) if "Make" in tg else ""; model = str(tg["Model"].value) if "Model" in tg else ""
    except Exception:
        pass
    return make, model


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _to_scene(lin, primaries="srgb"):
    M = output.rgb_to_xyz(output.SPACES[primaries]) if primaries in output.SPACES else output.rgb_to_xyz(primaries)
    return (lin.reshape(-1, 3) @ (XYZ_TO_BT2020 @ M).T.astype(np.float32)).reshape(lin.shape)


def _encoded(path, data):
    """Encoded images: gain-map HDR, JPEG XL, or anything Pillow reads. Returns img, report."""
    suf = Path(path).suffix.lower(); rep = dict(scene_estimated=True, ai_extended=False)
    if suf in (".jpg", ".jpeg", ".hdr") and ic.ultrahdr_check(data):
        try:
            hdr = ic.ultrahdr_decode(data)
            arr = np.asarray(hdr, np.float32)[..., :3]
            rep.update(source_class="hdr-gainmap", note="gain map decoded; linear HDR, assumed BT.2020/P3 per libultrahdr; 1.0 = 203 cd/m2")
            return _to_scene(arr, "p3"), rep
        except Exception as e:
            rep["gainmap_error"] = str(e)[:120]
    if suf == ".jxl":
        a = ic.jpegxl_decode(data)
        a = a[..., None].repeat(3, -1) if a.ndim == 2 else a[..., :3]
        if a.ndim == 4:  # animation: first frame
            a = a[0]
        if np.issubdtype(a.dtype, np.floating):
            rep.update(source_class="encoded-float", note="float JXL taken as linear BT.709 (colour metadata not exposed by the decoder: assumption)")
            return _to_scene(a.astype(np.float32)), rep
        rep.update(source_class="encoded-sdr", note="JXL integer decode, assumed sRGB (colour metadata not exposed by the decoder: assumption)")
        return _to_scene(_srgb_to_linear(a.astype(np.float32) / np.iinfo(a.dtype).max)), rep
    im = Image.open(io.BytesIO(data)); icc = im.info.get("icc_profile"); rep["icc"] = bool(icc)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im, np.float32) / 65535; a = np.repeat(a[..., None], 3, -1)
        rep.update(source_class="encoded-sdr", note="16-bit greyscale, assumed sRGB"); return _to_scene(_srgb_to_linear(a)), rep
    im = im.convert("RGB")
    if icc:  # through the embedded profile into a BT.2020 + sRGB-curve working space (8-bit Pillow path)
        try:
            dst = ImageCms.ImageCmsProfile(io.BytesIO(output.icc_profile_prim(BT2020_PRIM, "ColorForge BT.2020 working")))
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), dst, outputMode="RGB")
            rep.update(source_class="encoded-sdr", note="ICC to BT.2020; tone curve from the camera stays baked in")
            return _srgb_to_linear(np.asarray(im, np.float32) / 255), rep
        except Exception as e:
            rep["icc_error"] = str(e)[:120]
    rep.update(source_class="encoded-sdr", note="no usable ICC: assumed sRGB; tone curve from the camera stays baked in")
    return _to_scene(_srgb_to_linear(np.asarray(im, np.float32) / 255)), rep


def load_scene(path, denoiser=None, lens=True):
    """Returns img (H,W,3 float32 scene-linear BT.2020 D65), report dict."""
    path = Path(path); suf = path.suffix.lower()
    if suf in RAW_EXT:
        make, model = _exif(path)
        if suf == ".dng" and _is_own_dng(path):
            img, rep, _, _ = to_cf_scene(str(path), denoiser)
            rep.update(source_class="raw-own", scene_estimated=False, phone_profile=phone_profile(rep.get("make") or make, rep.get("model") or model))
            return img, rep
        try:
            cam = camera_raw.load(path, denoiser, lens)
            if cam is not None:
                img, rep = cam
                return img, dict(rep, make=rep.get("make") or make, model=rep.get("model") or model, source_class="raw-camera", scene_estimated=False, phone_profile=phone_profile(rep.get("make") or make, rep.get("model") or model))
            img, info = _libraw(path)
        except rawpy.LibRawError:  # e.g. DNG 1.7 LinearRaw with JPEG XL
            img, info = _linear_dng(path)
            return img, dict(info, file=path.name, source_class="raw-linear-dng", scene_estimated=False, phone_profile=phone_profile(info["make"], info["model"]))
        return img, dict(file=path.name, make=make, model=model, source_class="raw-libraw", scene_estimated=False, phone_profile=phone_profile(make, model), libraw=info,
                         note="Non-Bayer sensor: LibRaw's linear XYZ, camera white balance; no AI denoise or lens correction on this path")
    data = path.read_bytes(); img, rep = _encoded(path, data); rep.update(file=path.name, phone_profile=None)
    return img, rep
