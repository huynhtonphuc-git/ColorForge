"""E-6 stocks from the vector curves of manufacturer data sheets, and from MIT-licensed digitised data where the sheet's curves are bitmaps. Writes data/stocks/<id>.cfstock.json.

Each chart: a page box, then axes calibrated from the printed tick labels. Curves are named by their colour, or by the R, G, B letters printed beside them, or by content (sensitivity and dye peaks).
Every curve value is `datasheet`. Base density is not printed by any sheet, so it is the dye mix that gives the sheet's own Dmin, labelled `assumption`.
Usage: python digitise_sheets.py <datasheet folder>
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "data" / "stocks"
BOX_C, BOX_S, BOX_M, BOX_D = (55, 185, 292, 382), (318, 185, 548, 382), (55, 458, 292, 632), (305, 458, 548, 632)  # Fuji page layout, one page holds all four
LOGX = [-3.0, -2.0, -1.0, 0.0, 1.0]
SHEETS = {
    "ektachrome_e100": dict(name="Kodak Ektachrome E100", pdf="tate_E-4024_E100G.pdf", edition="Kodak E-4024 (09/2009), characteristic curve of the E100GX panel; Kodak's E100 sheet E-4000 (2018) prints the same 13 curves", iso=100, rms=8, exposure="daylight 1/100 s", charts=dict(C=(5, (60, 380, 295, 590)), S=(5, (330, 125, 578, 314)), D=(5, (335, 395, 578, 565)), M=(6, (40, 90, 290, 266)))),
    "ektachrome_e100g": dict(name="Kodak Ektachrome E100G", pdf="tate_E-4024_E100G.pdf", edition="Kodak E-4024 (09/2009)", iso=100, rms=8, exposure="daylight 1/100 s", charts=dict(C=(5, (60, 125, 295, 335)), S=(5, (330, 125, 578, 314)), D=(5, (335, 395, 578, 565)), M=(6, (40, 90, 290, 266)))),
    "astia_100f": dict(name="Fujichrome Astia 100F", pdf="125px_fuji_astia_100f_datasheet.pdf", edition="Fujifilm AF3-149E (2003)", iso=100, rms=7, exposure="daylight 1/50 s", charts=dict(C=(8, BOX_C), S=(8, BOX_S), D=(8, BOX_D), M=(8, BOX_M))),
    "velvia_100f": dict(name="Fujichrome Velvia 100F", pdf="125px_fuji_velvia_100f_datasheet.pdf", edition="Fujifilm AF3-148E (2003)", iso=100, rms=8, exposure="daylight 1/50 s", charts=dict(C=(8, (55, 108, 292, 308)), S=(8, (318, 108, 548, 312)), D=(8, (305, 385, 548, 556)), M=(8, (55, 385, 292, 556)))),
    "ektachrome_e100vs": dict(name="Kodak Ektachrome E100VS", pdf="125px_kodak_e163-Ektachrome_E100VS.pdf", edition="Kodak E-163 (09/2005)", iso=100, rms=11, exposure="daylight 1/100 s", charts=dict(C=(5, (55, 130, 292, 335)), S=(5, (330, 85, 578, 262)), D=(5, (55, 375, 292, 592)), M=(5, (330, 345, 578, 530)))),
    "ektachrome_epn": dict(name="Kodak Ektachrome 100 EPN", pdf="125px_kodak_e27-Ektachrome_100_EPN.pdf", edition="Kodak E-27 (07/2007)", iso=100, rms=11, exposure="daylight 1/100 s", charts=dict(C=(5, (55, 118, 292, 330)), S=(5, (330, 110, 578, 292)), D=(5, (55, 350, 292, 568)), M=(5, (330, 350, 582, 535)))),
}
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _paths(page, box):
    """Stroked curves inside the box, split at pen lifts. Grid lines are thin or straight."""
    x0, y0, x1, y1 = box; out = []
    for d in page.get_drawings():
        r = d["rect"]; w = d.get("width") or 0
        if d.get("type") not in ("s", "fs") or w < 0.6 or r.x0 < x0 or r.x1 > x1 or r.y0 < y0 or r.y1 > y1 or r.width < 18:
            continue
        seg = []; last = None
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]; new = [(b.x, b.y)]
            elif it[0] == "c":
                p = [np.array([q.x, q.y]) for q in it[1:5]]; a = it[1]; new = [tuple((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]) for t in np.linspace(0.05, 1, 16)]
            else:
                continue
            if last is None or abs(a.x - last[0]) + abs(a.y - last[1]) > 0.8:
                if len(seg) > 3:
                    out.append((np.array(seg), d.get("color")))
                seg = [(a.x, a.y)]
            seg += new; last = new[-1]
        if len(seg) > 3:
            out.append((np.array(seg), d.get("color")))
    # join pieces of one colour that touch, keep real curves
    return [(p[np.argsort(p[:, 0], kind="stable")], c) for p, c in out if np.ptp(p[:, 0]) > 18 and np.ptp(p[:, 1]) > 3]


def _axes(page, box, pts, log_x=False, log_y=False):
    """Straight lines from page points to data, from tick labels below and to the left of the curves."""
    x0, y0, x1, y1 = box; words = [w for w in page.get_text("words") if NUM.match(w[4]) and x0 <= w[0] and w[2] <= x1 and y0 <= w[1] and w[3] <= y1]; L = pts[:, 0].min(); B = pts[:, 1].max()
    def largest(items, key, tol):  # the tick labels are the biggest group sharing one line
        best = []
        for a in items:
            g = [b for b in items if abs(key(a) - key(b)) <= tol]
            if len(g) > len(best):
                best = g
        return best
    xs = [((w[0] + w[2]) / 2, float(w[4])) for w in largest([w for w in words if (w[1] + w[3]) / 2 > B - 4], lambda w: (w[1] + w[3]) / 2, 3)]; ys = [((w[1] + w[3]) / 2, float(w[4])) for w in largest([w for w in words if w[2] < L + 3], lambda w: w[2], 5)]
    if not log_x and len(xs) == 5 and all(v >= 0 for _, v in xs) and len({v for _, v in xs}) < 5:  # signs printed apart from the digits: -3 .. 1 from left to right
        xs = [(a, v) for (a, _), v in zip(sorted(xs), LOGX)]
    if not log_y and len(ys) == 5 and all(v >= 0 for _, v in ys) and len({v for _, v in ys}) < 5:  # sensitivity axis, 2 .. -2 from the top
        ys = [(a, v) for (a, _), v in zip(sorted(ys), (2.0, 1.0, 0.0, -1.0, -2.0))]
    fx = lambda v: np.log10(v) if log_x else v; fy = lambda v: np.log10(v) if log_y else v
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(f"axis labels not found: x {xs} y {ys}")
    xs = sorted(set(xs)); ys = sorted(set(ys)); px = np.polyfit([a for a, _ in xs], [fx(v) for _, v in xs], 1); py = np.polyfit([a for a, _ in ys], [fy(v) for _, v in ys], 1); rx = np.abs(np.polyval(px, [a for a, _ in xs]) - [fx(v) for _, v in xs]).max(); ry = np.abs(np.polyval(py, [a for a, _ in ys]) - [fy(v) for _, v in ys]).max()
    return px, py, float(rx), float(ry), len(xs), len(ys)


def _letters(page, box):
    x0, y0, x1, y1 = box; return {w[4]: ((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) for w in page.get_text("words") if w[4] in "RGB" and x0 <= w[0] <= x1 and y0 <= w[1] <= y1}


def _by_colour(paths, names):
    want = dict(R=(1, 0, 0), G=(0, .7, .3), B=(.2, .2, .7), C=(0, .7, .95), M=(.9, 0, .55), Y=(1, .95, 0)); out = {}
    for n in names:
        s = sorted(paths, key=lambda pc: sum((a - b) ** 2 for a, b in zip(pc[1], want[n]))); out[n] = s[0][0]
    return out


def _coloured(paths):
    return all(c and max(c) - min(c) > 0.3 for _, c in paths)


def chart(page, box, kind):
    paths = _paths(page, box)
    if not paths:
        raise ValueError(f"no curves in {kind}")
    px, py, rx, ry, nx, ny = _axes(page, box, np.vstack([p for p, _ in paths]), kind == "M", kind == "M")
    if kind in "CS":
        names = "RGB"
    elif kind == "D":
        names = "CMY"
    else:
        names = "RGB" if len(paths) >= 3 else "V"
    if names == "V":
        cur = {"V": max(paths, key=lambda pc: len(pc[0]))[0]}
    elif _coloured(paths):
        cur = _by_colour(paths, names)
    elif kind == "S":
        cur = dict(zip("RGB", [p for p, _ in sorted(paths, key=lambda pc: -pc[0][np.argmin(pc[0][:, 1]), 0])]))
    elif kind == "D":  # three narrowest peaks are the dyes; neutral is the broadest
        three = sorted(paths, key=lambda pc: pc[0][:, 1].min())[-3:] if len(paths) > 3 else paths
        cur = dict(zip("CMY", [p for p, _ in sorted(three, key=lambda pc: -pc[0][np.argmin(pc[0][:, 1]), 0])]))
    else:
        L = _letters(page, box); dist = lambda p, q: float(np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]).min()); cur = {}; free = [p for p, _ in paths]
        for n in sorted(L, key=lambda n: min(dist(p, L[n]) for p in free)):
            if n in "RGB" and free:
                b = min(free, key=lambda p: dist(p, L[n])); cur[n] = b; free = [p for p in free if p is not b]
        if len(cur) < 3 and kind == "M":  # the letters are not text on this sheet: the three layers' mean stands in for the visual curve
            fq = np.geomspace(max(p[:, 0].min() for p, _ in paths), min(p[:, 0].max() for p, _ in paths), 80); cur = {"V": np.c_[fq, np.mean([np.interp(fq, p[:, 0], p[:, 1]) for p, _ in paths], 0)]}
        elif len(cur) < 3:
            raise ValueError(f"{kind}: named {list(cur)} of 3 curves")
    data = {}
    for n, p in cur.items():
        x = np.polyval(px, p[:, 0]); y = np.polyval(py, p[:, 1]); o = np.argsort(x); x, y = x[o], y[o]; keep = np.r_[True, np.diff(x) > 1e-6]; x, y = x[keep], y[keep]; data[n] = np.c_[10 ** x, 10 ** y] if kind == "M" else np.c_[x, y]
    return data, dict(curves=len(paths), axis_err=(round(rx, 4), round(ry, 4)), ticks=(nx, ny))


def field(value, prov, src, note=None, **k):
    d = dict(value=value, provenance=prov, source=src); d.update(k)
    if note:
        d["note"] = note
    return d


def _js(a):
    return [[None if np.isnan(x) else round(float(x), 5) for x in r] for r in a]


def build(sid, c, folder):
    g = json.loads((OUT / "kodachrome_64.cfstock.json").read_text())["grids"]; LOGE = np.array(g["log_exposure"]); WL = np.array(g["wavelengths_nm"]); doc = pymupdf.open(Path(folder) / c["pdf"]); res = {}; info = {}
    for k, (pn, box) in c["charts"].items():
        res[k], info[k] = chart(doc[pn - 1], box, k)
    on = lambda d, names, grid: np.stack([np.interp(grid, d[n][:, 0], d[n][:, 1], left=np.nan, right=np.nan) for n in names], 1); src = c["edition"] + ", vector curves digitised by ColorForge (src/colorforge/digitise_sheets.py)"; C = res["C"]; M = res["M"]
    if "V" in M:
        mtf = M["V"]; note = "diffuse visual, one curve"
    else:
        fq = np.geomspace(max(M[n][0, 0] for n in "RGB"), min(M[n][-1, 0] for n in "RGB"), 80); mtf = np.c_[fq, sum(w * np.interp(fq, M[n][:, 0], M[n][:, 1]) for n, w in zip("RGB", (.25, .65, .10)))]; note = "R, G, B curves on the sheet; this is their visual mix 0.25/0.65/0.10 (mix is an assumption). Per-layer: see per_layer"
    P = dict(density_curves=field(_js(on(C, "RGB", LOGE)), "datasheet", src, f"integral Status A R,G,B; {c['exposure']}, E-6; log H in lux-s; null outside the drawn range", density_kind="status_a"),
             log_sensitivity=field(_js(on(res["S"], "RGB", WL)), "datasheet", src, "log10 sensitivity as printed; null outside the drawn range"), dye_density=field(_js(on(res["D"], "CMY", WL)), "datasheet", src, "C,M,Y spectral density as printed"),
             mtf=dict(field([dict(cycles_per_mm=[round(float(x), 4) for x in mtf[:, 0]], response=[round(float(x) / 100, 4) for x in mtf[:, 1]])], "datasheet", src, note), per_layer=None if "V" in M else {n: dict(cycles_per_mm=[round(float(x), 4) for x in M[n][:, 0]], response=[round(float(x) / 100, 4) for x in M[n][:, 1]]) for n in "RGB"}), rms_granularity=field(c["rms"], "datasheet", c["edition"], "diffuse RMS x1000, 48 um aperture, read at gross diffuse density 1.0"), iso=field(c["iso"], "datasheet", c["edition"]),
             dmin_status_a=field([round(float(C[n][-1, 1]), 3) for n in "RGB"], "datasheet", src, "right end of the drawn curves"))
    (OUT / f"{sid}.cfstock.json").write_text(json.dumps(dict(schema="colorforge.stock/0.2", id=sid, name=c["name"], type="positive", process="E-6", grids=g, primary_source_per_field={k: c["edition"] for k in P}, primary=P, alternates={}), indent=0)); return info


def from_sfl():
    """Sheets whose curves are bitmaps: the MIT-licensed digitisation in spectral_film_lut, with RMS from the sheet itself (that project's Provia value is 9; the sheet prints 8)."""
    done = {}
    for sid, src_id, name, iso, rms, ed in (("velvia_50_sheet", "velvia_50", "Fujichrome Velvia 50", 50, 9, "Fujifilm AF3-0221E2 (2007)"), ("provia_100f_sheet", "provia_100f", "Fujichrome Provia 100F", 100, 8, "Fujifilm AF3-036E (2000)"), ("ektachrome_100d_sheet", "ektachrome_100d", "Kodak Ektachrome 100D", 100, None, "Kodak H-1-5285")):
        j = json.loads((OUT / f"{src_id}.cfstock.json").read_text()); f = j["alternates"].get("spectral_film_lut") or j["primary"]; P = {k: f[k] for k in ("density_curves", "log_sensitivity", "dye_density", "mtf") if k in f}
        P["iso"] = dict(f.get("iso") or {}, value=iso, provenance="datasheet", source=ed)
        if rms is not None:
            P["rms_granularity"] = dict(value=rms, provenance="datasheet", source=ed, note="diffuse RMS x1000, 48 um aperture")
        else:
            P["rms_granularity"] = dict(value=8, provenance="assumption", source="E100G/E100GX sheet E-4024, nearest printed value", note="sheet H-1-5285 prints granularity curves, not one number")
        (OUT / f"{sid}.cfstock.json").write_text(json.dumps(dict(j, id=sid, name=name, primary=P, alternates={}, primary_source_per_field={k: "spectral_film_lut (MIT, Jan Lohse)" for k in P}), indent=0)); done[sid] = list(P)
    return done


if __name__ == "__main__":
    folder = sys.argv[1]
    for sid, c in SHEETS.items():
        try:
            print(sid, build(sid, c, folder), flush=True)
        except Exception as e:
            print(sid, "FAILED", type(e).__name__, str(e)[:220], flush=True)
    print(from_sfl())
