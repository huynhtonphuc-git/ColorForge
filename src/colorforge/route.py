"""Input routing: decide denoise, blur removal and enhancement from what the file really is (chart: docs/PIPELINE_ROUTING.md).

Facts come from the file, never from its extension alone: bit depth (white level), single frame vs stacked (mosaic vs LinearRaw /
already-demosaiced), phone vs camera (Make), the noise model. A plan is recorded in every render report.
"""
import math

PHONE_MAKES = ("xiaomi", "redmi", "samsung", "google", "apple", "huawei", "honor", "oneplus", "oppo", "vivo", "realme", "motorola", "sony mobile", "nothing", "htc", "lg", "hmd")


def denoise_strength(S, O):
    """Stronger denoise for noisier photos, from the photo's own NoiseProfile: SNR at 18 % signal. SNR 50 -> 0.40, SNR 10 -> 0.90
    (end points are choices tuned by eye and scan texture, not measurements). Lower strength keeps more real micro-texture."""
    snr = 0.18 / math.sqrt(max(S * 0.18 + O, 1e-12))
    t = (math.log(50) - math.log(max(snr, 1e-3))) / (math.log(50) - math.log(10))
    return round(min(0.95, max(0.30, 0.40 + 0.50 * t)), 3), round(snr, 1)


def plan(rep):
    """rep: capture report from ingest.load_scene. Returns the routing plan actually applied."""
    cls = rep.get("source_class"); make = str(rep.get("make") or "").lower(); bits = (rep.get("raw") or {}).get("bits") or (rep.get("libraw") or {}).get("bits")
    is_phone = any(make.startswith(m) for m in PHONE_MAKES)
    if cls == "raw-own":
        kind = "phone single-frame raw" if is_phone else "camera mosaic DNG"
    elif cls == "raw-linear-dng":
        kind = "stacked / linear phone raw"
    elif cls == "raw-camera":
        kind = "phone single-frame raw" if is_phone else "camera raw"
    elif cls == "raw-libraw":
        kind = "phone single-frame raw (LibRaw path)" if is_phone else "camera raw"
    elif cls in ("hdr-gainmap", "encoded-float"):
        kind = "HDR encoded"
    else:
        kind = "SDR encoded"
    # Enhancement is for input that is short of bits: single-frame low-bit raw and 8-bit encoded files. Stacked, high-bit raw and HDR files already have the data.
    enhance_ok = kind == "SDR encoded" or (kind in ("phone single-frame raw", "phone single-frame raw (LibRaw path)", "camera mosaic DNG", "camera raw") and bool(bits) and bits < 12)
    deblur = rep.get("phone_profile")  # only phones we measured; others: per-photo estimate is not built yet
    return dict(kind=kind, bits=bits, is_phone=is_phone, denoise=(rep.get("denoise") or {}).get("strength"), lens=rep.get("lens_correction"), phone_blur_removal=deblur,
                ai_enhancement=enhance_ok, ai_hdr_extension=None, scene_estimated=rep.get("scene_estimated"))
