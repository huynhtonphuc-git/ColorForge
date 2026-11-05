"""Render photos as Kodachrome 64 slides -> one Ultra HDR / ISO 21496-1 JPEG per photo (Display P3).

Chain: any input (ingest.py: phone DNG with PMRID denoise, camera raw, DNG 1.7 JPEG XL, gain-map JPEG, JPEG XL...) -> CF-Scene -> per-photo auto exposure -> film layer exposure (log H)
-> interimage (nonlinear, spatial) -> acutance (scatter + Nelson CSF, um) -> dye-cloud grain -> GPU 3D LUT
(dyes, projector, viewing) -> gamut compression in P3 -> HDR JPEG (SDR base = lightbox appearance,
HDR = slide colorimetry anchored at mid-grey).
Usage: python render_film.py <out_dir> <stock.cfstock.json> [options] file.dng ...
Also writes a small <stem>_preview.jpg (scene | film, 1600 px) and render_report.json for review.
"""
import argparse, json, time
from pathlib import Path

import colour
import cv2
import numpy as np

import acutance
import acutance2
import film
import grain
import interimage
import output
import route
import ingest
from cf_dng import srgb_preview


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("stock"); ap.add_argument("files", nargs="+")
    ap.add_argument("--ev", type=float, default=0.0, help="exposure offset (EV) added to the per-photo auto exposure")
    ap.add_argument("--interimage", type=float, default=0.6, help="interimage strength (1 = Fuji E-6 patent mid range; 0.6 calibrated to Kodachrome scans; 0 = off)")
    ap.add_argument("--grain", type=float, default=0.5, help="grain strength (1 = Selwyn upper bound from RMS 10; 0.5 calibrated to scans; 0 = off)")
    ap.add_argument("--denoise-strength", type=float, default=None, help="fixed AI denoise strength; default: routed from each photo's noise")
    ap.add_argument("--grey-pack", type=float, default=0.0, help="film grey balance strength: 0 = none (default, user decision: 1 looks too yellow), 1 = greys on the scans' measured neutral")
    ap.add_argument("--wb", default="asshot", choices=["asshot", "auto"], help="asshot: trust the camera (default, user decision); auto: our residual estimate")
    ap.add_argument("--temp", type=float, default=0.0, help="white balance slider: + warmer, - cooler (-100..100, 100 = 1 stop)")
    ap.add_argument("--tint", type=float, default=0.0, help="tint slider: + magenta, - green (-100..100, 100 = 1 stop)")
    ap.add_argument("--no-deblur", action="store_true", help="do not remove the phone blur (Wiener, measured MTF)")
    ap.add_argument("--no-acutance", action="store_true"); ap.add_argument("--no-denoise", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = time.time(); st = film.Stock(a.stock); M, mfit = film.fit_exposure_matrix(st, film.daylight(5500))
    lamp = colour.sd_blackbody(3200, film.SHAPE).values
    lut_sdr = film.bake_lut(st, lamp, viewing=film.VIEWING["lightbox"]); lut_hdr = film.bake_lut(st, lamp, viewing=film.VIEWING["hdr"])
    # HDR anchor (assumption): an 18 % grey card has the same luminance in HDR as in SDR; clear film then rises above SDR white
    off = float(np.log10(10 / st.iso)); g = np.full((1, 1, 3), off, np.float32)
    Yg = lambda lut: float(film.apply_lut_cpu(g, lut, "bt2020")[0, 0] @ [0.2627, 0.6780, 0.0593])
    hdr_scale = Yg(lut_sdr) / Yg(lut_hdr)
    _p = film.grey_balance(lut_sdr, off) * a.grey_pack; pack = np.array([_p[0], 0.0, _p[1]], np.float32)  # filter pack R, G, B in log H
    v_k = grain.unit_visual_density(st, film.CMF[:, 1], lamp)
    setup = dict(stock=st.name, exposure_matrix=M.round(5).tolist(), matrix_fit=mfit, assumptions=film.ASSUMPTIONS,
                 sdr_viewing=film.VIEWING["lightbox"], hdr_viewing=film.VIEWING["hdr"], hdr_clear_film_vs_sdr_white=round(hdr_scale, 3), filter_pack_logH=pack.round(3).tolist(),
                 interimage=dict(interimage.KODACHROME_64, strength=a.interimage), acutance=None if a.no_acutance else acutance2.KODACHROME_64, denoise_strength=a.denoise_strength, phone_deblur=None if a.no_deblur else acutance2.REDMI_TURBO4PRO,
                 grain=dict(grain.KODACHROME_64, strength=a.grain), unit_dye_visual_density=v_k.round(3).tolist(), setup_s=round(time.time() - t, 1))
    print(json.dumps(setup, indent=1)); reps = []
    if not a.cpu:
        import gpu
    dn = None
    if not a.no_denoise:
        from denoise import RawDenoiser
        _dn = RawDenoiser("CPU" if a.cpu else "AUTO"); setup["denoise_backend"] = _dn.backend
        dn = lambda c, pat, S, O: _dn(c, pat, S, O, strength=a.denoise_strength if a.denoise_strength is not None else route.denoise_strength(S, O)[0])
    I = np.eye(3, dtype=np.float32)
    for i, f in enumerate(a.files):
      try:
        t0 = time.time(); img, rep = ingest.load_scene(f, dn); t1 = time.time()
        wb_rep = None
        if a.wb == "auto" and str(rep.get("source_class", "")).startswith("raw"):  # encoded files carry a baked look: leave them
            gains, wb_rep = film.auto_white(img, (rep.get("colour") or {}).get("cct_K")); img = img * gains
        if a.temp or a.tint:
            img = img * film.manual_white(a.temp, a.tint); wb_rep = dict(wb_rep or {}, temp=a.temp, tint=a.tint)
        ev, exp_rep = film.auto_exposure(img, st); ev += a.ev; img = img * np.float32(2 ** ev)
        prev = srgb_preview(img, 0.0, 1600)
        logH = film.scene_to_logH(img, M, st); del img; upx = acutance.um_per_px(logH.shape); ii_rep = gr_rep = None
        if a.interimage > 0:
            logH, ii_rep = interimage.apply(st, logH, dict(interimage.KODACHROME_64, strength=a.interimage), upx)
        if not a.no_acutance:
            logH, ac_rep = acutance2.apply(st, logH, acutance2.KODACHROME_64, upx, phone=None if a.no_deblur or rep.get("phone_profile") != "REDMI_TURBO4PRO" else acutance2.REDMI_TURBO4PRO)
        if a.grain > 0:
            logH, gr_rep = grain.apply(st, logH, dict(grain.KODACHROME_64, strength=a.grain), upx, v_k, seed=i)
        t2 = time.time(); eq = (0.18 * 10 ** (logH + pack - off)).astype(np.float32); del logH  # equivalent exposure, identity matrix
        if a.cpu:
            run = lambda lut: film.apply_lut_cpu(film.scene_to_logH(eq, I, st), lut, "bt2020")
        else:
            run = lambda lut: gpu.film_lut(eq, I, off, lut["bt2020"], lut["grid"])
        sdr = np.clip(output.compress_gamut(output.from_bt2020(run(lut_sdr), "p3")), 0, 1)
        hdr = output.compress_gamut(output.from_bt2020(run(lut_hdr), "p3")) * np.float32(hdr_scale)
        t3 = time.time(); stem = Path(f).stem; name = f"{stem}_kodachrome64_hdr.jpg"
        nbytes = output.write_hdr_jpeg(sdr, hdr, out / name)
        small = cv2.resize(sdr, (1600, int(sdr.shape[0] * 1600 / sdr.shape[1])), interpolation=cv2.INTER_AREA)  # preview: P3 values shown as sRGB
        cv2.imwrite(str(out / f"{stem}_preview.jpg"), cv2.cvtColor(np.concatenate([prev, (output.srgb_oetf(small) * 255 + 0.5).astype(np.uint8)], 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        Y = sdr.reshape(-1, 3) @ np.array([0.2290, 0.6917, 0.0793], np.float32)  # Display P3 luminance
        r = dict(file=Path(f).name, source_class=rep.get("source_class"), scene_estimated=rep.get("scene_estimated"), phone_profile=rep.get("phone_profile"), out=name, bytes=nbytes, ev_applied=round(ev, 2), exposure=exp_rep, white_balance=wb_rep, interimage=ii_rep, grain=gr_rep,
                 denoise=rep.get("denoise"), decode_s=round(t1 - t0, 1), physics_s=round(t2 - t1, 1), lut_s=round(t3 - t2, 2), write_s=round(time.time() - t3, 1),
                 sdr_Y_percentiles={k: round(float(v), 4) for k, v in zip(["p1", "p50", "p99"], np.percentile(Y, [1, 50, 99]))},
                 hdr_p999_vs_sdr_white=round(float(np.percentile(hdr.max(-1), 99.9)), 2))
        r["plan"] = route.plan(rep); reps.append(r); print(json.dumps(r), flush=True)
      except Exception as e:  # one bad file must not stop the run
        reps.append(dict(file=Path(f).name, error=f"{type(e).__name__}: {str(e)[:200]}")); print(json.dumps(reps[-1]), flush=True)
    (out / "render_report.json").write_text(json.dumps(dict(setup=setup, renders=reps), indent=1, default=str))


if __name__ == "__main__":
    main()
