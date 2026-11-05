"""Digitise Kodachrome 25 (E-55 06/2009 p3) and Kodachrome 200 (p5) from the vector paths of Kodak's PDF. Measures Kodak's curves; the PDF stays out of the repo.

Pages 3, 4 and 5 use one chart layout at different page positions. Each axis is calibrated on page 4, whose curves were validated earlier (data/kodak_e55, Q60 target):
a straight line from page points to data units (log-spaced axes included, since they are drawn in log). The other pages reuse it after a shift equal to the mean offset
of the chart's tick labels. The Kodachrome 25 log H axis is printed 1.0 higher than the other two (-2..2 against -3..1). Check: p5 MTF against the separate E-88 drawing.

Usage: python digitise_e55.py <e55-2009_06.pdf> <e88-2009_06.pdf> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pymupdf

CHARTS = {  # page-point box holding each chart's curves on page 4, used to find its paths and labels
    "HD": (55, 110, 300, 320), "SENS": (55, 380, 300, 560), "DYE": (330, 120, 560, 320), "MTF": (325, 375, 575, 560)}
MIN_ITEMS = {"HD": 8, "SENS": 8, "DYE": 8, "MTF": 12}


def sample(d):
    """Points along a stroked path, Bezier segments sampled."""
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            p = [np.array([q.x, q.y]) for q in it[1:5]]
            for t in np.linspace(0, 1, 12):
                pts.append(tuple((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]))
    a = np.array(pts); return a[np.argsort(a[:, 0], kind="stable")]


def curves(page, box):
    x0, y0, x1, y1 = box; out = []
    for d in page.get_drawings():
        r = d["rect"]; n = len(d["items"])
        if n >= 8 and d.get("type") == "s" and r.x0 >= x0 and r.x1 <= x1 and r.y0 >= y0 and r.y1 <= y1 and r.width > 20 and d.get("fill") is None and all(i[0] in "lc" for i in d["items"]):
            out.append(sample(d))
    return out


APPROX = {3: dict(HD=(15, 18), SENS=(12, 16), DYE=(18, 14), MTF=(17, 14)), 5: dict(HD=(12, 0), SENS=(12, 0), DYE=(12, 0), MTF=(12, 0))}  # read from the tick labels, refined below


def nums(page):
    return np.array([(w[0], w[1]) for w in page.get_text("words") if w[4].replace(".", "").isdigit()])


def page_shift(a, b, pn, chart):
    """Exact (dx, dy) moving page a's chart onto page b's: each tick label on a is paired with the label nearest its expected place on b."""
    x0, y0, x1, y1 = CHARTS[chart]; ex, ey = APPROX[pn][chart]; A = nums(a); A = A[(A[:, 0] > x0 - 25) & (A[:, 0] < x1 + 5) & (A[:, 1] > y0 - 5) & (A[:, 1] < y1 + 5)]; B = nums(b); d = []
    for q in A:
        r = np.abs(B - q - (ex, ey)).max(1)
        if r.min() < 3:
            d.append(B[r.argmin()] - q)
    d = np.array(d); return float(np.median(d[:, 0])), float(np.median(d[:, 1])), float(d[:, 0].std()), float(d[:, 1].std()), len(d)


def fit_axes(paths, refs):
    """Best common straight-line maps for all curves of one chart."""
    ax, bx = np.polyfit(np.r_[[p[:, 0].min() for p in paths], [p[:, 0].max() for p in paths]], np.r_[[r[0, 0] for r in refs], [r[-1, 0] for r in refs]], 1)[::-1]
    X, Y = [], []
    for p, r in zip(paths, refs):
        X += list(p[:, 1]); Y += list(np.interp(ax + bx * p[:, 0], r[:, 0], r[:, 1]))
    by, ay = np.polyfit(X, Y, 1); res = np.polyval([by, ay], X) - np.array(Y)
    return ax, bx, ay, by, float(np.percentile(np.abs(res), 95))


def order(chart, cs):
    """Label curves by content."""
    if chart == "HD":
        return dict(zip("RGB", sorted(cs, key=lambda a: -a[0, 1])))
    if chart == "SENS":
        return dict(zip("RGB", sorted(cs, key=lambda a: -a[np.argmax(a[:, 1]), 0])))
    if chart == "DYE":
        c3 = sorted(cs, key=lambda a: a[:, 1].max())[:3]
        return dict(zip("CMY", sorted(c3, key=lambda a: -a[np.argmax(a[:, 1]), 0])))
    return {"V": max(cs, key=len)}


def clean(a):
    a = a[np.argsort(a[:, 0], kind="stable")]; return a[np.r_[True, np.diff(a[:, 0]) > 1e-3]]


def main(e55, e88, out):
    out = Path(out); doc = pymupdf.open(e55); ref_dir = out; p4 = doc[3]; res = {}; load = lambda f: clean(np.loadtxt(f, delimiter=",", skiprows=1))
    files = dict(HD=[ref_dir / f"e55-2009_06_p4_HD_d{i}.csv" for i in (3, 4, 5)], SENS=[ref_dir / f"e55-2009_06_p4_SENS_d{i}.csv" for i in (18, 19, 20)],
                 DYE=[ref_dir / f"e55-2009_06_p4_DYE_d{i}.csv" for i in (49, 50, 51, 52)], MTF=[next(ref_dir.glob("e88-2009_06_p4_MTF_*.csv"))])
    for chart, box in CHARTS.items():
        paths = curves(p4, box); refs = [load(f) for f in files[chart]]
        if chart == "MTF":
            paths = [max(paths, key=len)]
            refs = [np.c_[np.log10(refs[0][:, 0]), np.log10(np.maximum(refs[0][:, 1], 1e-3))]]  # log axes are straight on the page
        pl = order(chart, [np.c_[c[:, 0], -c[:, 1]] for c in paths]); pl = {k: np.c_[v[:, 0], -v[:, 1]] for k, v in pl.items()}; rl = order(chart, refs); keys = list(pl)
        ax, bx, ay, by, err = fit_axes([pl[k] for k in keys], [rl[k] for k in keys]); print(chart, "p4 fit, 95 % residual", round(err, 4), "paths", len(paths), flush=True)
        for pn, tag in ((3, "p3"), (5, "p5")):
            page = doc[pn - 1]; dx, dy, sx, sy, n = page_shift(p4, page, pn, chart); print(" ", tag, "shift", round(dx, 2), round(dy, 2), "label scatter", round(sx, 2), round(sy, 2), "labels", n, flush=True)
            cs = curves(page, (box[0] + dx - 3, box[1] + dy - 3, box[2] + dx + 3, box[3] + dy + 3))
            if chart == "MTF":
                cs = [max(cs, key=len)]
            data = []
            for c in cs:
                d = np.c_[ax + bx * (c[:, 0] - dx), ay + by * (c[:, 1] - dy)]
                if chart == "HD" and pn == 3:
                    d[:, 0] += 1.0  # this page's axis runs -2..2
                data.append(clean(d))
            res[(tag, chart)] = order(chart, data)
            for k, d in res[(tag, chart)].items():
                if chart == "MTF":
                    d = np.c_[10 ** d[:, 0], 10 ** d[:, 1]]; res[(tag, chart)][k] = d
                np.savetxt(out / f"e55-2009_06_{tag}_{chart}_{k}.csv", d, delimiter=",", fmt="%.4f", header="cyc_per_mm,response_pct" if chart == "MTF" else "x,y", comments="")
    # independent check: E-88 p5 is a separate drawing of KL 200's MTF, already known to equal p4
    kl = load(next(ref_dir.glob("e88-2009_06_p4_MTF_*.csv"))); m = res[("p5", "MTF")]["V"]; xs = np.geomspace(3, 70, 30)
    print("MTF p5 vs E-88: max |log10| diff", round(float(np.abs(np.log10(np.interp(xs, m[:, 0], m[:, 1])) - np.log10(np.interp(xs, kl[:, 0], kl[:, 1]))).max()), 4))
    for tag, film in (("p3", "K25"), ("p5", "K200")):
        hd = res[(tag, "HD")]; print(film, {k: dict(left=[round(hd[k][0, 0], 3), round(hd[k][0, 1], 2)], dmin=round(hd[k][-1, 1], 2), right=round(hd[k][-1, 0], 2), logH_D1=round(float(np.interp(-1, -hd[k][:, 1], hd[k][:, 0])), 3)) for k in "RGB"}); m = res[(tag, "MTF")]["V"]
        print(" mtf", [round(float(np.interp(v, m[:, 0], m[:, 1])), 1) for v in (5, 10, 20, 40, 60)])


if __name__ == "__main__":
    main(*sys.argv[1:4])
