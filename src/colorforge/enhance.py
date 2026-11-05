"""AI micro-contrast. Runs at the end of Input Adapt on any film, off by default (restore.py).

Real-ESRGAN Compact (realesr-general-x4v3, Xintao Wang, BSD-3; third_party/RealESRGAN_compact). The architecture is re-implemented below and only the weights are loaded.
AI rule (CLAUDE.md): the model never sets colour or tone. Its result is used only as a brightness ratio, high-passed at 40 um on the film, so broad tone and all colour stay with
the input. The returned field doubles as the change mask. A restoration model can invent plausible texture: tests and the mask are there so this can be checked.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cf_dng import BT2020_TO_SRGB

WEIGHTS = Path(__file__).resolve().parents[2] / "third_party" / "RealESRGAN_compact" / "realesr-general-x4v3.pth"
CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ColorForge" / "ai_cache"
MICRO_STRENGTH = 0.6; SPLIT_UM = 40.0; CLIP_LOG = 0.2
# Dev laptop, full frame, final detail against CPU f32: Iris Xe f16 28 s, corr 0.987-0.9985, p99.9 difference 0.005-0.009 log before strength (film grain is about 0.02);
# Iris Xe f32 50 s, identical; CPU 98 s. Use "f32" here if the small difference ever matters.
GPU_PRECISION = "f16"


class _Compact(nn.Module):
    """SRVGGNetCompact: 32 feature layers, 4x. Same layer order as the published weights."""

    def __init__(self, feat=64, conv=32, up=4):
        super().__init__(); b = [nn.Conv2d(3, feat, 3, 1, 1), nn.PReLU(feat)]
        for _ in range(conv):
            b += [nn.Conv2d(feat, feat, 3, 1, 1), nn.PReLU(feat)]
        self.body = nn.ModuleList(b + [nn.Conv2d(feat, 3 * up * up, 3, 1, 1), nn.PixelShuffle(up)]); self.up = up

    def forward(self, x):
        y = x
        for m in self.body:
            y = m(y)
        return y + F.interpolate(x, scale_factor=self.up, mode="nearest")


def _encode(lin):
    x = np.clip(lin, 0, None); x = x / (1 + x / 4)  # same soft highlight as the inspection preview; the model expects display-like input
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055).astype(np.float32)


def _decode(v):
    x = np.where(v <= 0.04045, v / 12.92, ((np.maximum(v, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return x / np.maximum(1 - x / 4, 1e-3)


class MicroContrast:
    """Any GPU OpenVINO sees is tried first, then the CPU; each must pass a self-check against PyTorch."""

    def __init__(self, device="AUTO", tile=256, overlap=16):
        self.tile, self.ov = tile, overlap; net = _Compact(); net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["params"]); net.eval(); self.net = net; self.cm = None
        self.backend = "torch-cpu"; self.run = lambda x: net(torch.from_numpy(x)).detach().numpy()
        if torch.cuda.is_available() and device in ("AUTO", "CUDA"):  # NVIDIA with a CUDA build of PyTorch (not tested on the dev laptop)
            net = net.cuda(); self.backend = f"torch-cuda: {torch.cuda.get_device_name(0)}"; self.run = lambda x: net(torch.from_numpy(x).cuda()).detach().cpu().numpy()
            return
        self.notes = []
        try:
            import openvino as ov
            self.ovm = ov; self.core = ov.Core(); devs = self.core.available_devices
            for dev in (["GPU", "CPU"] if device == "AUTO" else [device]):
                if dev in devs and self._compile(dev):
                    break
        except Exception as e:
            self.notes.append(str(e)[:80])
        if self.cm is None:
            self.backend += f" (openvino failed: {self.notes})"

    def _compile(self, dev):
        size = self.tile + 2 * self.ov
        try:
            CACHE.mkdir(parents=True, exist_ok=True); xml = CACHE / f"esrgan_compact_{size}.xml"
            if not xml.exists():
                self.ovm.save_model(self.ovm.convert_model(self.net, example_input=torch.zeros(1, 3, size, size)), xml)
            # The input size must be fixed: with an open size the Iris Xe driver (30.0.101.1660) returns wrong numbers without any error.
            m = self.core.read_model(xml); m.reshape([1, 3, size, size])
            cm = self.core.compile_model(m, dev, {"INFERENCE_PRECISION_HINT": GPU_PRECISION if dev == "GPU" else "f32", "PERFORMANCE_HINT": "THROUGHPUT", "CACHE_DIR": str(CACHE)})
            # Every device must reproduce PyTorch on a test tile before it is trusted, so a bad driver falls back instead of silently damaging photos.
            yy, xx = np.mgrid[0:size, 0:size] / size; rng = np.random.default_rng(0); x = np.stack([0.2 + 0.5 * xx, 0.3 + 0.4 * yy, 0.5 - 0.3 * xx * yy], 0) + 0.02 * rng.standard_normal((3, size, size))
            x = np.clip(x, 0, 1).astype(np.float32)[None]; err = float(np.abs(cm(x)[0] - self.net(torch.from_numpy(x)).detach().numpy()).mean())
            if err > 0.003:
                self.notes.append(f"{dev}: failed self-check, error {err:.4f}")
                return False
            self.cm = cm; self.check_error = err; self.dev = dev; self.backend = f"openvino-{dev}: {self.core.get_property(dev, 'FULL_DEVICE_NAME')}"; self.run = lambda x: self.cm(x)[0]
            return True
        except Exception as e:
            self.notes.append(f"{dev}: {str(e)[:80]}"); self.cm = None
            return False

    def _net(self, enc):
        """enc (H, W, 3) -> model output resampled back to (H, W, 3), tiled with overlap. Tiles run in parallel when OpenVINO is in use."""
        H, W, _ = enc.shape; t, o = self.tile, self.ov; ny, nx = -(-H // t), -(-W // t); s = t + 2 * o
        pad = np.pad(enc, ((o, ny * t + o - H), (o, nx * t + o - W), (0, 0)), mode="reflect"); out = np.empty((ny * t, nx * t, 3), np.float32)

        def put(y, x, r):
            r = cv2.resize(r.transpose(1, 2, 0), (s, s), interpolation=cv2.INTER_AREA)  # 4x result back to the file's own resolution
            out[y * t:(y + 1) * t, x * t:(x + 1) * t] = r[o:o + t, o:o + t]
        tile = lambda y, x: np.ascontiguousarray(pad[y * t:y * t + s, x * t:x * t + s].transpose(2, 0, 1)[None], np.float32)
        if self.cm is None:
            for y in range(ny):
                for x in range(nx):
                    put(y, x, self.run(tile(y, x))[0])
            return out[:H, :W]
        q = self.ovm.AsyncInferQueue(self.cm, self.cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")); q.set_callback(lambda req, ud: put(ud[0], ud[1], req.get_output_tensor(0).data[0]))
        for y in range(ny):
            for x in range(nx):
                q.start_async({0: tile(y, x)}, (y, x))
        q.wait_all()
        return out[:H, :W]

    def _net_safe(self, enc):
        try:
            return self._net(enc)
        except Exception as e:  # an old GPU driver can fail mid-run: finish on the CPU instead of losing the render
            if self.cm is not None and getattr(self, "dev", "") == "GPU" and self._compile("CPU"):
                self.backend += f" (GPU failed: {str(e)[:60]})"; return self._net(enc)
            raise

    def __call__(self, scene, um_per_px):
        """scene: exposed, scene-linear BT.2020. Returns detail (H, W): high-passed log10 brightness change. Its magnitude is the AI change mask."""
        srgb = (scene.reshape(-1, 3) @ BT2020_TO_SRGB.T.astype(np.float32)).reshape(scene.shape); enc = _encode(srgb); W = np.array([0.2126, 0.7152, 0.0722], np.float32)
        Yi = np.maximum(srgb.clip(0, None) @ W, 1e-4); Ya = np.maximum(_decode(np.clip(self._net_safe(enc), 0, 1)) @ W, 1e-4)
        d = np.log10(Ya) - np.log10(Yi); d = np.clip(d, -1, 1)
        sig = SPLIT_UM / um_per_px; hp_of = lambda a: a - cv2.GaussianBlur(a, (0, 0), sig, borderType=cv2.BORDER_REFLECT); hp = hp_of(d)
        # The model sharpens edges but also smooths fine texture (measured: it lowered 12.5 um contrast by 3 %). Micro-contrast takes only the part that
        # raises contrast already present in the photo: same sign as the input's own local detail. Then high-pass again so low frequencies stay untouched.
        keep = cv2.GaussianBlur((hp * hp_of(np.log10(Yi)) > 0).astype(np.float32), (0, 0), 0.7)
        return np.clip(hp_of(hp * keep), -CLIP_LOG, CLIP_LOG).astype(np.float32)
