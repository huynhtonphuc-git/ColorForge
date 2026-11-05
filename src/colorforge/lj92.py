"""Minimal lossless JPEG (ITU T.81 process 14, SOF3) decoder for DNG tiles. Pure Python; used to verify fast decoders."""
import numpy as np


class _Bits:
    def __init__(self, data, pos):
        self.d = data; self.p = pos; self.acc = 0; self.n = 0

    def bit(self):
        if self.n == 0:
            b = self.d[self.p]; self.p += 1
            if b == 0xFF:
                nb = self.d[self.p]
                if nb == 0x00:
                    self.p += 1
                else:  # marker: feed zeros
                    self.p -= 1; b = 0
            self.acc = b; self.n = 8
        self.n -= 1
        return (self.acc >> self.n) & 1

    def bits(self, k):
        v = 0
        for _ in range(k):
            v = (v << 1) | self.bit()
        return v


def _build_huff(counts, symbols):
    # returns dict (length, code) -> symbol
    table = {}; code = 0; k = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            table[(length, code)] = symbols[k]; k += 1; code += 1
        code <<= 1
    return table


def decode(data):
    data = bytes(data); i = 2; huff = {}; comps = []; pred = 1; pt = 0
    while True:
        assert data[i] == 0xFF, "marker expected"
        m = data[i + 1]; L = int.from_bytes(data[i + 2:i + 4], "big"); seg = data[i + 4:i + 2 + L]
        if m == 0xC3:
            prec = seg[0]; H = int.from_bytes(seg[1:3], "big"); W = int.from_bytes(seg[3:5], "big"); nc = seg[5]
            comps = [seg[6 + 3 * j] for j in range(nc)]
        elif m == 0xC4:
            j = 0
            while j < len(seg):
                tc_th = seg[j]; counts = list(seg[j + 1:j + 17]); n = sum(counts)
                huff[tc_th & 0x0F] = _build_huff(counts, list(seg[j + 17:j + 17 + n])); j += 17 + n
        elif m == 0xDA:
            ns = seg[0]; sel = [seg[2 + 2 * j] >> 4 for j in range(ns)]
            pred = seg[1 + 2 * ns]; pt = seg[3 + 2 * ns] & 0x0F
            start = i + 2 + L
            break
        i += 2 + L
    br = _Bits(data, start); out = np.zeros((H, W, nc), np.int32); init = 1 << (prec - pt - 1)
    tabs = [huff[s] for s in sel]
    for y in range(H):
        for x in range(W):
            for c in range(nc):
                t = tabs[c]; code = 0; length = 0
                while True:
                    code = (code << 1) | br.bit(); length += 1
                    s = t.get((length, code))
                    if s is not None:
                        break
                if s == 0:
                    diff = 0
                elif s == 16:
                    diff = 32768
                else:
                    v = br.bits(s)
                    diff = v if v >= (1 << (s - 1)) else v - (1 << s) + 1
                if y == 0 and x == 0:
                    p = init
                elif y == 0:
                    p = out[0, x - 1, c]
                elif x == 0:
                    p = out[y - 1, 0, c]
                else:
                    a = out[y, x - 1, c]; b = out[y - 1, x, c]; cc = out[y - 1, x - 1, c]
                    p = {1: a, 2: b, 3: cc, 4: a + b - cc, 5: a + ((b - cc) >> 1), 6: b + ((a - cc) >> 1), 7: (a + b) >> 1}[pred]
                out[y, x, c] = (p + diff) & 0xFFFF
    return out, dict(H=H, W=W, nc=nc, prec=prec, pred=pred)
