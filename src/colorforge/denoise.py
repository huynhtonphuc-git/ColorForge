"""Raw Bayer denoise with PMRID (Wang et al., ECCV 2020, Megvii, Apache-2.0), adapted per photo.

The network architecture below is re-implemented from third_party/PMRID/models/net_torch.py (Apache-2.0, Megvii);
only the weights file is loaded (torch.load weights_only=True: tensors, no code). Credit: github.com/megvii-research/PMRID.

Per-photo adaptation (k-sigma transform): PMRID expects noise var(x_DN) = k*x_DN + s on a V = 959 DN scale, mapped to
its training anchor (ISO 1600 of an OPPO Reno 10x). The DNG NoiseProfile gives var = S*x + O on a 0..1 scale, so
k = S*V, s = O*V^2 -- exact for every photo, no ISO table needed.

AI rule (CLAUDE.md): output = low frequencies of the input + high frequencies of the network, and a change map in
units of the input noise sigma. Runs on the raw mosaic after black subtraction, before lens-shading gain.
Backends, in order: PyTorch CUDA (NVIDIA), OpenVINO GPU (Intel), OpenVINO CPU, PyTorch CPU. docs/COMPUTE_BACKENDS.md.
"""
from collections import OrderedDict
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

WEIGHTS = Path(__file__).resolve().parents[2] / "third_party" / "PMRID" / "models" / "torch_pretrained.ckp"
CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ColorForge" / "ai_cache"
V = 959.0
K_A = 0.0005995267 * 1600 + 0.00868861  # anchor k (ISO 1600, PMRID run_benchmark.py)
S_A = 7.11772e-7 * 1600 ** 2 + 6.514934e-4 * 1600 + 0.11492713  # anchor read variance
INP_SCALE = 256.0


def _conv(i, o, k, s, p, sep=False, relu=False):
    m = OrderedDict()
    if sep:
        m["depthwise"] = nn.Conv2d(i, i, k, s, p, groups=i, bias=False); m["pointwise"] = nn.Conv2d(i, o, 1, 1, 0, bias=True)
    else:
        m["conv"] = nn.Conv2d(i, o, k, s, p, bias=True)
    if relu:
        m["relu"] = nn.ReLU()
    return nn.Sequential(m)


class _Enc(nn.Module):
    def __init__(self, i, mid, o, stride=1):
        super().__init__()
        self.conv1 = _conv(i, mid, 5, stride, 2, True, True); self.conv2 = _conv(mid, o, 5, 1, 2, True, False)
        self.proj = nn.Identity() if stride == 1 and i == o else _conv(i, o, 3, stride, 1, True, False); self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.proj(x))


def _enc_stage(i, o, n):
    return nn.Sequential(_Enc(i, o // 4, o, 2), *[_Enc(o, o // 4, o, 1) for _ in range(n - 1)])


class _DecBlock(nn.Module):
    def __init__(self, i, o, k=3):
        super().__init__(); self.conv0 = _conv(i, o, k, 1, k // 2, True, True); self.conv1 = _conv(o, o, k, 1, k // 2, True, False)

    def forward(self, x):
        return self.conv1(self.conv0(x)) + x


class _DecStage(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.decode_conv = _DecBlock(i, i); self.upsample = nn.ConvTranspose2d(i, o, 2, 2, 0); self.proj_conv = _conv(skip, o, 3, 1, 1, True, True)

    def forward(self, x, skip):
        return self.upsample(self.decode_conv(x)) + self.proj_conv(skip)


class PMRID(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = _conv(4, 16, 3, 1, 1, False, True)
        self.enc1 = _enc_stage(16, 64, 2); self.enc2 = _enc_stage(64, 128, 2); self.enc3 = _enc_stage(128, 256, 4); self.enc4 = _enc_stage(256, 512, 4)
        self.encdec = _conv(512, 64, 3, 1, 1, True, True)
        self.dec1 = _DecStage(64, 256, 64); self.dec2 = _DecStage(64, 128, 32); self.dec3 = _DecStage(32, 64, 32); self.dec4 = _DecStage(32, 16, 16)
        self.out0 = _DecBlock(16, 16); self.out1 = _conv(16, 4, 3, 1, 1, False, False)

    def forward(self, inp):
        c0 = self.conv0(inp); c1 = self.enc1(c0); c2 = self.enc2(c1); c3 = self.enc3(c2); c4 = self.enc4(c3)
        x = self.dec4(self.dec3(self.dec2(self.dec1(self.encdec(c4), c3), c2), c1), c0)
        return inp + self.out1(self.out0(x))


class RawDenoiser:
    def __init__(self, device="AUTO", tile=512, overlap=32):
        self.tile, self.ov = tile, overlap
        net = PMRID(); net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)); net.eval()
        self.backend = "torch-cpu"; self.run = lambda x: net(torch.from_numpy(x)).detach().numpy()
        if device == "AUTO" and not torch.cuda.is_available():  # discrete GPU if there is one; on Iris Xe the CPU is faster for this model (3.1 vs 4.0 s, 12.5 MP)
            try:
                import openvino as ov
                c = ov.Core(); device = "GPU" if "GPU" in c.available_devices and "dGPU" in c.get_property("GPU", "FULL_DEVICE_NAME") else "CPU"
            except Exception:
                device = "CPU"
        if device in ("GPU", "AUTO") and torch.cuda.is_available():  # NVIDIA with a CUDA build of PyTorch (not tested on the dev laptop)
            net = net.cuda(); self.backend = f"torch-cuda: {torch.cuda.get_device_name(0)}"; self.run = lambda x: net(torch.from_numpy(x).cuda()).detach().cpu().numpy()
        elif device in ("GPU", "CPU"):
            try:
                import openvino as ov
                core = ov.Core()
                if device in core.available_devices:
                    CACHE.mkdir(parents=True, exist_ok=True); xml = CACHE / f"pmrid_{tile}.xml"
                    if not xml.exists():
                        ov.save_model(ov.convert_model(net, example_input=torch.zeros(1, 4, tile, tile)), xml)
                    cm = core.compile_model(core.read_model(xml), device, {"INFERENCE_PRECISION_HINT": "f32", "CACHE_DIR": str(CACHE)})  # kernels are cached between launches
                    self.run = lambda x: cm(x)[0]; self.backend = f"openvino-{device}: {core.get_property(device, 'FULL_DEVICE_NAME')}"
            except Exception as e:  # keep torch CPU
                self.backend += f" (openvino failed: {str(e)[:80]})"

    def _net_full(self, planes):
        """planes (4,h,w) in network units -> network output, tiled with overlap (static tile shape for OpenVINO)."""
        _, h, w = planes.shape; t, o = self.tile, self.ov; step = t - 2 * o
        ny, nx = -(-h // step), -(-w // step)  # tiles needed; padded size = n*step + 2*overlap
        pad = np.pad(planes, ((0, 0), (o, ny * step + o - h), (o, nx * step + o - w)), mode="reflect")
        out = np.zeros_like(pad)
        for y in range(0, pad.shape[1] - t + 1, step):
            for x in range(0, pad.shape[2] - t + 1, step):
                r = self.run(np.ascontiguousarray(pad[None, :, y:y + t, x:x + t], np.float32))[0]
                out[:, y + o:y + t - o, x + o:x + t - o] = r[:, o:t - o, o:t - o]
        return out[:, o:o + h, o:o + w]

    def __call__(self, cfa01, pattern, S, O, keep_lowfreq_sigma=4.0, strength=1.0):
        """cfa01: raw mosaic, black-subtracted, /(white-black), before gain maps. pattern: DNG CFAPattern (2x2).
        S, O: NoiseProfile terms (green). Returns denoised mosaic, change map (|delta|/noise sigma), info."""
        flip = {(0, 1, 1, 2): (False, False), (2, 1, 1, 0): (True, True), (1, 0, 2, 1): (False, True), (1, 2, 0, 1): (True, False)}[tuple(pattern)]
        x = cfa01[::-1] if flip[0] else cfa01; x = x[:, ::-1] if flip[1] else x  # -> RGGB
        H, W = x.shape; planes = x.reshape(H // 2, 2, W // 2, 2).transpose(1, 3, 0, 2).reshape(4, H // 2, W // 2)
        k, s = S * V, O * V * V; ck = K_A / k; cb = (s / k ** 2 - S_A / K_A ** 2) * K_A
        net_in = (np.clip(planes, 0, 1) * V * ck + cb) / V * INP_SCALE
        pred = (self._net_full(net_in.astype(np.float32)) / INP_SCALE * V - cb) / ck / V
        # AI rule: keep the input's low frequencies, take only the network's high frequencies
        lp = lambda a: np.stack([cv2.GaussianBlur(np.float32(c), (0, 0), keep_lowfreq_sigma, borderType=cv2.BORDER_REFLECT) for c in a])
        lo = lp(planes); hf_ai, hf_in = pred - lp(pred), planes - lo
        outp = lo + strength * hf_ai + (1 - strength) * hf_in  # strength < 1 keeps some real micro-texture (and some noise)
        out = outp.reshape(2, 2, H // 2, W // 2).transpose(2, 0, 3, 1).reshape(H, W)
        out = out[::-1] if flip[0] else out; out = out[:, ::-1] if flip[1] else out
        sigma = np.sqrt(np.maximum(S * np.clip(cfa01, 0, None) + O, 1e-12))
        change = np.abs(out - cfa01) / sigma
        return out.astype(np.float32), change.astype(np.float16), dict(model="PMRID (ECCV 2020, Apache-2.0)", backend=self.backend,
                                                                       k=round(k, 4), read_var=round(s, 4), anchor_k=round(K_A, 4), lowfreq_sigma_px=keep_lowfreq_sigma, strength=strength)
