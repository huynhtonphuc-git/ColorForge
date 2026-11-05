"""Compiled CPU kernels (Numba): the fast path for the DNG decoder and per-pixel lookups. lj92.py stays as the readable reference, and tests compare the two.

Do not call the parallel kernels from more than one Python thread at a time: Numba's default threading layer does not allow it.
The app's job worker is the only caller.
"""
import numpy as np
from numba import njit, prange


@njit(nogil=True, cache=True)
def _scan(d, pos, H, W, nc, init, pred, mincode, maxcode, valptr, vals, sel, out):
    """Lossless-JPEG entropy decode + prediction. out: flat uint16, H * W * nc. Huffman tables are canonical, one row per table id."""
    acc = np.int64(0); n = 0; size = d.size; pos = np.int64(pos); s = np.int64(0); p = np.int64(0); diff = np.int64(0)
    for y in range(H):
        for x in range(W):
            for c in range(nc):
                t = np.int64(sel[c]); code = np.int64(0); length = 0
                while True:  # one Huffman code, bit by bit
                    if n == 0:
                        b = np.int64(d[pos]) if pos < size else np.int64(0); pos += 1
                        if b == 0xFF:
                            nb = np.int64(d[pos]) if pos < size else np.int64(0xD9)
                            if nb == 0:
                                pos += 1
                            else:  # marker: feed zeros
                                pos -= 1; b = 0
                        acc = b; n = 8
                    n -= 1; code = (code << 1) | ((acc >> n) & 1); length += 1
                    if length > 16:
                        return False
                    if maxcode[t, length] >= 0 and code <= maxcode[t, length] and code >= mincode[t, length]:
                        s = np.int64(vals[t, np.int64(valptr[t, length]) + code - np.int64(mincode[t, length])])
                        break
                if s == 0:
                    diff = np.int64(0)
                elif s == 16:
                    diff = np.int64(32768)
                else:
                    v = np.int64(0)
                    for _ in range(s):
                        if n == 0:
                            b = np.int64(d[pos]) if pos < size else np.int64(0); pos += 1
                            if b == 0xFF:
                                nb = np.int64(d[pos]) if pos < size else np.int64(0xD9)
                                if nb == 0:
                                    pos += 1
                                else:
                                    pos -= 1; b = 0
                            acc = b; n = 8
                        n -= 1; v = (v << 1) | ((acc >> n) & 1)
                    diff = v if v >= (1 << (s - 1)) else v - (1 << s) + 1
                i = y * W * nc + x * nc + c  # flat index
                if y == 0 and x == 0:
                    p = np.int64(init)
                elif y == 0:
                    p = np.int64(out[i - nc])
                elif x == 0:
                    p = np.int64(out[i - W * nc])
                else:
                    A = np.int64(out[i - nc]); B = np.int64(out[i - W * nc]); C = np.int64(out[i - W * nc - nc])
                    if pred == 1:
                        p = A
                    elif pred == 2:
                        p = B
                    elif pred == 3:
                        p = C
                    elif pred == 4:
                        p = A + B - C
                    elif pred == 5:
                        p = A + ((B - C) >> 1)
                    elif pred == 6:
                        p = B + ((A - C) >> 1)
                    else:
                        p = (A + B) >> 1
                out[i] = (p + diff) & 0xFFFF
    return True


def lj92_decode(data):
    """Decode one lossless-JPEG tile. Returns uint16 (H, W*nc) and an info dict, same values as lj92.decode."""
    data = bytes(data); i = 2; mincode = np.zeros((4, 17), np.int32); maxcode = np.full((4, 17), -1, np.int32); valptr = np.zeros((4, 17), np.int32); vals = np.zeros((4, 256), np.uint8)
    while True:
        assert data[i] == 0xFF, "marker expected"
        m = data[i + 1]; L = int.from_bytes(data[i + 2:i + 4], "big"); seg = data[i + 4:i + 2 + L]
        if m == 0xC3:
            prec = seg[0]; H = int.from_bytes(seg[1:3], "big"); W = int.from_bytes(seg[3:5], "big"); nc = seg[5]
        elif m == 0xC4:
            j = 0
            while j < len(seg):
                t = seg[j] & 0x0F; counts = seg[j + 1:j + 17]; total = sum(counts); vals[t, :total] = list(seg[j + 17:j + 17 + total]); code = 0; k = 0
                for ln in range(1, 17):
                    if counts[ln - 1]:
                        valptr[t, ln] = k; mincode[t, ln] = code; k += counts[ln - 1]; code += counts[ln - 1]; maxcode[t, ln] = code - 1
                    code <<= 1
                j += 17 + total
        elif m == 0xDA:
            ns = seg[0]; sel = np.array([seg[2 + 2 * j] >> 4 for j in range(ns)], np.int32); pred = seg[1 + 2 * ns]; pt = seg[3 + 2 * ns] & 0x0F; start = i + 2 + L
            break
        i += 2 + L
    out = np.zeros(H * W * nc, np.uint16)
    if not _scan(np.frombuffer(data, np.uint8), start, H, W, nc, 1 << (prec - pt - 1), int(pred), mincode, maxcode, valptr, vals, sel, out):
        raise ValueError("corrupt lossless-JPEG data")
    return out.reshape(H, W * nc), dict(H=H, W=W, nc=nc, prec=prec, pred=pred)


@njit(parallel=True, nogil=True, cache=True)
def _table(x, x0, step, tab, out):
    H, W = x.shape; n = tab.size
    for i in prange(H):
        for j in range(W):
            t = (x[i, j] - x0) / step
            if t <= 0:
                out[i, j] = tab[0]
            elif t >= n - 1:
                out[i, j] = tab[n - 1]
            else:
                k = int(t); f = t - k
                out[i, j] = tab[k] + f * (tab[k + 1] - tab[k])


def table(x, x0, step, tab):
    """Linear interpolation on a uniform grid, like np.interp but parallel. x is 2-D float32."""
    out = np.empty(x.shape, np.float32)
    _table(np.ascontiguousarray(x, np.float32), float(x0), float(step), np.ascontiguousarray(tab, np.float32), out)
    return out


class Curves:
    """A stock's dye-amount curves as uniform tables, forward (log H -> amount) and inverse. Forward values equal np.interp's within 1e-3; the inverse uses the same reversed np.interp, on a fine grid."""
    N = 8192

    def __init__(self, stock):
        self.x0 = float(stock.loge[0]); self.x1 = float(stock.loge[-1]); self.step = (self.x1 - self.x0) / (self.N - 1); g = np.linspace(self.x0, self.x1, self.N); self.a = []; self.inv = []; self.amax = []
        for c in range(3):
            curve = stock.amounts[:, c]; self.a.append(np.interp(g, stock.loge, curve).astype(np.float32)); self.amax.append(float(curve.max()))
            lo, hi = float(curve.min()), float(curve.max()); ag = np.linspace(lo, hi, self.N)
            self.inv.append((lo, (hi - lo) / (self.N - 1), np.interp(ag, curve[::-1], stock.loge[::-1]).astype(np.float32)))

    def amount(self, logH, c):
        return table(logH, self.x0, self.step, self.a[c])

    def logH(self, amount, c):
        lo, step, tab = self.inv[c]
        return table(amount, lo, step, tab)


def curves(stock):
    if not hasattr(stock, "_curves"):
        stock._curves = Curves(stock)
    return stock._curves
