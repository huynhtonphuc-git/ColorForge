"""ColorForge GPU layer (WebGPU via wgpu-py): device setup, tiling and kernels.

Design rules (Iris Xe first, any GPU later):
- one shared device; pick a discrete GPU if present, else integrated, Vulkan/D3D12 before OpenGL
- every buffer <= MAX_BUFFER (OpenGL caps a storage binding at 128 MiB), images processed in row tiles
- each kernel has a NumPy reference; tests require agreement (tests/test_gpu.py)
Heavy AI models do not run here (see denoise.py and docs/COMPUTE_BACKENDS.md); this module is for the display chain, and later blurs and grain.
"""
import os

import numpy as np
import wgpu

MAX_BUFFER = 96 * 2 ** 20
_dev = None

LUT_SHADER = """
struct P { m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, g: vec4<f32> };   // rows of M, (offset, lo, hi, n)
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

fn fetch(i: u32, j: u32, k: u32, n: u32) -> vec3<f32> {
    let b = ((i * n + j) * n + k) * 3u;
    return vec3<f32>(lut[b], lut[b + 1u], lut[b + 2u]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x + id.y * 65535u * 256u;
    if (px * 3u >= arrayLength(&src)) { return; }
    let rgb = vec3<f32>(src[3u * px], src[3u * px + 1u], src[3u * px + 2u]);
    // D1: exposure matrix, then log10 exposure in lux-s
    let e = max(vec3<f32>(dot(p.m0.xyz, rgb), dot(p.m1.xyz, rgb), dot(p.m2.xyz, rgb)), vec3<f32>(1e-6)) / 0.18;
    let logh = log2(e) * 0.30102999566 + vec3<f32>(p.g.x);
    // D4 + D6: trilinear 3D LUT over log H
    let n = u32(p.g.w);
    let t = (clamp(logh, vec3<f32>(p.g.y), vec3<f32>(p.g.z)) - vec3<f32>(p.g.y)) / (p.g.z - p.g.y) * (p.g.w - 1.0);
    let i0 = vec3<u32>(min(floor(t), vec3<f32>(p.g.w - 2.0)));
    let f = t - vec3<f32>(i0);
    var c = vec3<f32>(0.0);
    for (var a = 0u; a < 2u; a++) { for (var b = 0u; b < 2u; b++) { for (var d = 0u; d < 2u; d++) {
        let w = select(1.0 - f.x, f.x, a == 1u) * select(1.0 - f.y, f.y, b == 1u) * select(1.0 - f.z, f.z, d == 1u);
        c += w * fetch(i0.x + a, i0.y + b, i0.z + d, n);
    } } }
    dst[3u * px] = c.x; dst[3u * px + 1u] = c.y; dst[3u * px + 2u] = c.z;
}
"""


DEVELOP_SHADER = """
struct P { r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, delta: vec4<f32>, g: vec4<f32>, z: vec4<f32>, s: vec4<f32> };
// r0-r2: BT.2020 -> Display P3. delta: log H offsets. g: (lo, hi, n, -). z: zone k low, mid, high and contrast k. s: saturation k, vibrance, gamut scale, -.
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
const LUMA = vec3<f32>(0.2290, 0.6917, 0.0793);

fn fetch(i: u32, j: u32, k: u32, n: u32) -> vec3<f32> {
    let b = ((i * n + j) * n + k) * 3u;
    return vec3<f32>(lut[b], lut[b + 1u], lut[b + 2u]);
}
fn zone(L: f32, s: f32, c: f32, w: f32) -> f32 {
    let t = (L - c) / w;
    return s * exp(-0.5 * t * t) * (L - c);
}
fn compress1(d: f32) -> f32 {
    if (d < 0.8) { return d; }
    let x = (d - 0.8) / p.s.z;
    return 0.8 + p.s.z * x / pow(1.0 + pow(x, 1.2), 1.0 / 1.2);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x + id.y * 65535u * 256u;
    if (px * 3u >= arrayLength(&src)) { return; }
    let logh = vec3<f32>(src[3u * px], src[3u * px + 1u], src[3u * px + 2u]) + p.delta.xyz;
    let n = u32(p.g.z);
    let t = (clamp(logh, vec3<f32>(p.g.x), vec3<f32>(p.g.y)) - vec3<f32>(p.g.x)) / (p.g.y - p.g.x) * (p.g.z - 1.0);
    let i0 = vec3<u32>(min(floor(t), vec3<f32>(p.g.z - 2.0)));
    let f = t - vec3<f32>(i0);
    var c = vec3<f32>(0.0);
    for (var a = 0u; a < 2u; a++) { for (var b = 0u; b < 2u; b++) { for (var e = 0u; e < 2u; e++) {
        let w = select(1.0 - f.x, f.x, a == 1u) * select(1.0 - f.y, f.y, b == 1u) * select(1.0 - f.z, f.z, e == 1u);
        c += w * fetch(i0.x + a, i0.y + b, i0.z + e, n);
    } } }
    var q = vec3<f32>(dot(p.r0.xyz, c), dot(p.r1.xyz, c), dot(p.r2.xyz, c));
    if (p.z.x != 0.0 || p.z.y != 0.0 || p.z.z != 0.0) {   // studio.tone_zones
        let L = log2(max(dot(q, LUMA), 1e-5) / 0.18);
        q *= pow(2.0, zone(L, p.z.x, -3.5, 1.6) + zone(L, p.z.y, -0.5, 1.2) + zone(L, p.z.z, 1.2, 1.0));
    }
    if (p.z.w != 0.0) { q *= pow(max(dot(q, LUMA), 1e-5) / 0.18, p.z.w); }   // studio.global_contrast
    if (p.s.x != 1.0 || p.s.y != 0.0) {   // studio.saturate
        let Y = dot(q, LUMA); var k = p.s.x;
        if (p.s.y != 0.0) {
            let hi = max(q.x, max(q.y, q.z)); let lo = min(q.x, min(q.y, q.z)); let sat = clamp((hi - lo) / max(hi, 1e-5), 0.0, 1.0);
            k *= 1.0 + p.s.y * (1.0 - sat) * (1.0 - sat);
        }
        q = vec3<f32>(Y) + k * (q - vec3<f32>(Y));
    }
    let ach = max(q.x, max(q.y, q.z)); let safe = max(abs(ach), 1e-6);   // output.compress_gamut
    let dd = (vec3<f32>(ach) - q) / safe;
    q = vec3<f32>(ach) - vec3<f32>(compress1(dd.x), compress1(dd.y), compress1(dd.z)) * safe;
    dst[3u * px] = q.x; dst[3u * px + 1u] = q.y; dst[3u * px + 2u] = q.z;
}
"""
_res = {}  # pipeline and LUT buffers kept between calls


def adapters():
    """All GPUs that WebGPU can see, for reports and for choosing. Works the same on Intel, NVIDIA, AMD and Apple."""
    return [dict(name=a.info.get("device"), backend=a.info.get("backend_type"), type=a.info.get("adapter_type")) for a in wgpu.gpu.enumerate_adapters_sync()]


def device():
    """Discrete GPU first, then integrated; Vulkan or Metal, then D3D12. COLORFORGE_GPU=<part of the name> forces one."""
    global _dev
    if _dev is None:
        rank = {"DiscreteGPU": 0, "IntegratedGPU": 1}; brank = {"Vulkan": 0, "D3D12": 1, "Metal": 0}
        ads = [a for a in wgpu.gpu.enumerate_adapters_sync() if a.info.get("adapter_type") in rank]
        want = os.environ.get("COLORFORGE_GPU", "").lower(); ads = [a for a in ads if want in str(a.info.get("device")).lower()] or ads
        if not ads:
            raise RuntimeError("no WebGPU GPU adapter")
        a = min(ads, key=lambda a: (rank[a.info["adapter_type"]], brank.get(a.info.get("backend_type"), 2)))
        lim = {k.replace("_", "-"): v for k, v in a.limits.items()}
        _dev = a.request_device_sync(required_limits={k: lim[k] for k in ("max-storage-buffer-binding-size", "max-buffer-size")})
        _dev.cf_name = f"{a.info['device']} [{a.info['backend_type']}]"
    return _dev


def develop(logH, delta, lut, grid, to_p3, low=0.0, mid=0.0, high=0.0, contrast=0.0, saturation=0.0, vibrance=0.0, gamut_scale=1.0):
    """Whole display chain in one pass: log H -> film LUT -> Display P3 -> calibrate sliders -> gamut compression.
    Same maths as Studio.develop_cpu, kept equal by tests/test_film.py. Row tiles stay under MAX_BUFFER."""
    dev = device(); lo, hi, n = grid
    if "develop" not in _res:
        _res["develop"] = dev.create_compute_pipeline(layout="auto", compute={"module": dev.create_shader_module(code=DEVELOP_SHADER), "entry_point": "main"})
    pipe = _res["develop"]; key = ("lut", id(lut))
    if key not in _res:
        _res[key] = (lut, dev.create_buffer_with_data(data=np.ascontiguousarray(lut, np.float32).ravel(), usage=wgpu.BufferUsage.STORAGE))  # keeps lut alive so its id stays valid
    lb = _res[key][1]; U = np.zeros((7, 4), np.float32); U[:3, :3] = to_p3; U[3, :3] = delta; U[4, :3] = (lo, hi, n)
    U[5] = (low / 200, mid / 200, high / 200, contrast / 200); U[6, :3] = (1 + saturation / 100, vibrance / 100, gamut_scale)
    ub = dev.create_buffer_with_data(data=U, usage=wgpu.BufferUsage.UNIFORM)
    H, W, _ = logH.shape; rows = max(1, MAX_BUFFER // (W * 12)); out = np.empty(logH.shape, np.float32)
    for y in range(0, H, rows):
        tile = np.ascontiguousarray(logH[y:y + rows], np.float32); npx = tile.shape[0] * W
        sb = dev.create_buffer_with_data(data=tile.ravel(), usage=wgpu.BufferUsage.STORAGE)
        db = dev.create_buffer(size=tile.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate((sb, db, lb, ub))])
        groups = -(-npx // 256); enc = dev.create_command_encoder(); p = enc.begin_compute_pass()
        p.set_pipeline(pipe); p.set_bind_group(0, bg); p.dispatch_workgroups(min(groups, 65535), -(-groups // 65535)); p.end()
        dev.queue.submit([enc.finish()])
        out[y:y + rows] = np.frombuffer(dev.queue.read_buffer(db), np.float32).reshape(tile.shape)
    return out


def film_lut(img, M, offset, lut, grid):
    """img (H,W,3) scene BT.2020 float32 -> LUT output (H,W,3). Row tiles keep buffers under MAX_BUFFER."""
    dev = device(); lo, hi, n = grid
    pipe = getattr(dev, "_cf_lut_pipe", None)
    if pipe is None:
        pipe = dev._cf_lut_pipe = dev.create_compute_pipeline(layout="auto", compute={"module": dev.create_shader_module(code=LUT_SHADER), "entry_point": "main"})
    lb = dev.create_buffer_with_data(data=np.ascontiguousarray(lut, np.float32).ravel(), usage=wgpu.BufferUsage.STORAGE)
    P = np.zeros((4, 4), np.float32); P[:3, :3] = M; P[3] = (offset, lo, hi, n)
    ub = dev.create_buffer_with_data(data=P, usage=wgpu.BufferUsage.UNIFORM)
    H, W, _ = img.shape; rows = max(1, MAX_BUFFER // (W * 12)); out = np.empty_like(img, dtype=np.float32)
    for y in range(0, H, rows):
        tile = np.ascontiguousarray(img[y:y + rows], np.float32); npx = tile.shape[0] * W
        sb = dev.create_buffer_with_data(data=tile.ravel(), usage=wgpu.BufferUsage.STORAGE)
        db = dev.create_buffer(size=tile.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": b}} for i, b in enumerate((sb, db, lb, ub))])
        groups = -(-npx // 256); enc = dev.create_command_encoder(); p = enc.begin_compute_pass()
        p.set_pipeline(pipe); p.set_bind_group(0, bg); p.dispatch_workgroups(min(groups, 65535), -(-groups // 65535)); p.end()
        dev.queue.submit([enc.finish()])
        out[y:y + rows] = np.frombuffer(dev.queue.read_buffer(db), np.float32).reshape(tile.shape)
    return out
