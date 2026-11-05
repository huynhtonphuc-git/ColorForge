"""ColorForge desktop app: local server (127.0.0.1 only) + browser window. Standard library only.

Heavy work runs one job at a time on a worker thread, and the page polls /api/state. Preview requests run straight away.
Usage: python server.py [--port 8765] [--no-browser]   (or env\\run_colorforge.bat)
"""
import argparse, io, json, os, queue, shutil, subprocess, sys, tempfile, threading, time, traceback, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "src" / "colorforge"))
OUT = ROOT / "output" / "studio"; INBOX = Path(tempfile.gettempdir()) / "colorforge_inbox"
EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw", ".3fr", ".fff", ".rwl", ".iiq", ".jxl", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
FILMS = [
    dict(id="k14", name="Kodachrome 64", process="K-14", era="1974–2009", ready=True, note="Kodak datasheet, checked against real slides", swatch=["#b3261e", "#e8b23a", "#2b4f8c"]),
    dict(id="k14legacy", name="Kodachrome Legacy", process="K-14 + look", era="published", ready=True, note="Kodachrome 64 physics with a grade calibrated and fused from publicly shared film pictures. A look, not film data", swatch=["#a4161a", "#d98a1f", "#0f3f6e"]),
    dict(id="k14digital", name="Kodachrome Digital", process="K-14 + look", era="digital style", ready=True, note="Kodachrome 64 physics with a grade calibrated from publicly shared digital pictures. A look, not film data", swatch=["#bf3a2b", "#e8a92e", "#3f7fa6"]),
    dict(id="velvia50legacy", name="Velvia 50 Legacy", process="E-6 + look", era="published", ready=True, note="Velvia 50 physics with a grade calibrated and fused from publicly shared slide scans. A look, not film data", swatch=["#b3124a", "#6fa62e", "#0f55a8"]),
    dict(id="velvia50digital", name="Velvia 50 Digital", process="E-6 + look", era="digital style", ready=True, note="Velvia 50 physics with a grade calibrated from publicly shared digital pictures. A look, not film data", swatch=["#d41f5b", "#8dbb35", "#1a70c4"]),
    dict(id="provialegacy", name="Provia Legacy", process="E-6 + look", era="published", ready=True, note="Provia 100F physics with a grade calibrated and fused from publicly shared slide scans. A look, not film data", swatch=["#b52a26", "#a79a37", "#27699d"]),
    dict(id="proviadigital", name="Provia Digital", process="E-6 + look", era="digital style", ready=True, note="Provia 100F physics with a grade calibrated from publicly shared digital pictures. A look, not film data", swatch=["#cc3a33", "#c2ad3e", "#2f78b5"]),
    dict(id="k25", name="Kodachrome 25", process="K-14", era="1974–2002", ready=True, note="Kodak's own curves, dyes, MTF and RMS 9 (E-55). Rendering not yet compared with scans", swatch=["#b02a22", "#e6b53e", "#27508f"]),
    dict(id="k200", name="Kodachrome 200", process="K-14", era="1986–2007", ready=True, note="Kodak's own curves, dyes, MTF and RMS 16 (E-55). Rendering not yet compared with scans", swatch=["#a93a2c", "#deb04a", "#35598c"]),
    dict(id="velvia50", name="Velvia 50", process="E-6", era="1990–", ready=True, note="Fuji sheet curves, digitised by spectral_film_lut (MIT). RMS 9 from the sheet. Interimage shared with Kodachrome: assumption", swatch=['#c2185b', '#7cb342', '#1565c0']),
    dict(id="velvia100f", name="Velvia 100F", process="E-6", era="2003–", ready=True, note="Fuji sheet AF3-148E, vector curves. RMS 8. Interimage shared with Kodachrome: assumption", swatch=['#c62858', '#7fae44', '#1a66b5']),
    dict(id="provia", name="Provia 100F", process="E-6", era="2000–", ready=True, note="Fuji sheet curves, digitised by spectral_film_lut (MIT). RMS 8 from the sheet. Interimage shared with Kodachrome: assumption", swatch=['#c62828', '#8d9e3c', '#2e6da4']),
    dict(id="astia", name="Astia 100F", process="E-6", era="2003–", ready=True, note="Fuji sheet AF3-149E, vector curves. RMS 7. Interimage shared with Kodachrome: assumption", swatch=['#b84a43', '#94a04e', '#44739f']),
    dict(id="e100", name="Ektachrome E100", process="E-6", era="2018–", ready=True, note="Kodak E-4024 vector curves, which Kodak's 2018 E100 sheet repeats. RMS 8. Interimage shared with Kodachrome: assumption", swatch=['#b63b35', '#a3a44a', '#2f78a8']),
    dict(id="e100g", name="Ektachrome E100G", process="E-6", era="2005–2012", ready=True, note="Kodak E-4024, vector curves. RMS 8. Interimage shared with Kodachrome: assumption", swatch=['#b0413e', '#a9a24b', '#2f7fa8']),
    dict(id="e100vs", name="Ektachrome E100VS", process="E-6", era="1999–2012", ready=True, note="Kodak E-163, vector curves. RMS 11. Interimage shared with Kodachrome: assumption", swatch=['#c1272d', '#8ea83c', '#1f6fb0']),
    dict(id="epn", name="Ektachrome EPN", process="E-6", era="to 2009", ready=True, note="Kodak E-27, vector curves. RMS 11. Interimage shared with Kodachrome: assumption", swatch=['#a84a42', '#a39f55', '#3d7e9f']),
    dict(id="e100d", name="Ektachrome 100D", process="E-6", era="2000s", ready=True, note="Kodak 5285 curves, digitised by spectral_film_lut (MIT). RMS is the nearest printed value: assumption. Interimage: assumption", swatch=['#ad4641', '#a2a452', '#3581a6']),
]
_SHEET = dict(velvia50legacy=("Fuji AF3-0221E2", 9), velvia50digital=("Fuji AF3-0221E2", 9), provialegacy=("Fuji AF3-036E", 8), proviadigital=("Fuji AF3-036E", 8), k14legacy=("Kodak E-55", 10), k14digital=("Kodak E-55", 10), k14=("Kodak E-55", 10), k25=("Kodak E-55", 9), k200=("Kodak E-55", 16), velvia50=("Fuji AF3-0221E2", 9), velvia100f=("Fuji AF3-148E", 8), provia=("Fuji AF3-036E", 8), astia=("Fuji AF3-149E", 7), e100=("Kodak E-4024", 8), e100g=("Kodak E-4024", 8), e100vs=("Kodak E-163", 11), epn=("Kodak E-27", 11), e100d=("Kodak 5285", None))
_PROFILES = json.loads((ROOT / "data" / "stocks" / "film_physics.json").read_text()) if (ROOT / "data" / "stocks" / "film_physics.json").exists() else {}
for _f in FILMS:  # what each card starts from, and where each part comes from
    _sheet, _rms = _SHEET[_f["id"]]; _bid = {"velvia50legacy": "velvia50", "velvia50digital": "velvia50", "provialegacy": "provia", "proviadigital": "provia"}.get(_f["id"], _f["id"]); _e6 = _bid in _PROFILES
    _f["defaults"] = dict(interimage=_PROFILES[_bid]["interimage"] if _e6 else 0.6, grain=0.5, micro=1.0, clarity=0.0, camera_sharp=False)
    _f["tags"] = dict(micro="film at 100", grain=(f"2x RMS {_rms} at 50" if _rms else "2x film at 50") if _e6 else f"RMS {_rms} at 50")
    _f["sources"] = dict(sheet=_sheet, interimage="assumption, E-6 patent middle" if _e6 else "assumption, tuned on scans", grain=(f"{_sheet}, RMS {_rms}" if _rms else "assumption, RMS 8") if _e6 else f"RMS {_rms} at the middle")
for _f in FILMS:
    if _f["id"] in ("velvia50legacy", "velvia50digital", "provialegacy", "proviadigital"):
        _f["defaults"].update(micro=0.6); _f["tags"]["micro"] = "card default at 60"; _f["sources"]["sheet"] = _f["sources"]["sheet"] + ", then a fitted look"
    if _f["id"] in ("k14legacy", "k14digital"):
        _f["defaults"].update(micro=0.6, grain=1.0); _f["tags"] = dict(micro="card default at 60", grain="2x RMS 10 at 100"); _f["sources"].update(sheet="Kodak E-55, then a fitted look", grain="2x RMS 10")
FORMATS = [
    dict(id="jpeg_hdr", name="HDR JPEG", detail="Gain map, Display P3. Falls back to SDR on any screen"),
    dict(id="jpeg", name="JPEG", detail="8-bit Display P3, universal"),
    dict(id="jxl", name="JPEG XL", detail="16-bit lossless Display P3"),
    dict(id="dng", name="DNG", detail="16-bit linear, keeps HDR range, for further editing"),
]

state = dict(ready=False, busy=False, engine=None, message="Warming up the engines", error=None, file=None, generation=0, physics=None, plan=None, capture=None, seconds={}, auto_ev=None, exposure_info=None, exports=[], device="starting", films=FILMS, formats=FORMATS)
studio = None; jobs = queue.Queue(); lock = threading.Lock(); thumbs = {}; extra_files = []


def set_state(**k):
    with lock:
        state.update(k)


def clean(o):
    """JSON-safe copy of report values (numpy scalars and tuples)."""
    return json.loads(json.dumps(o, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def capture_info(s):
    r = s.rep; cap = r.get("capture") or {}; H, W = s.scene.shape[:2]
    return clean(dict(make=r.get("make"), model=r.get("model"), width=W, height=H, iso=cap.get("iso"), exposure_s=cap.get("exposure_s"), fnumber=cap.get("fnumber"), focal_mm=cap.get("focal_mm"),
                      white_k=(r.get("colour") or {}).get("cct_K"), source_class=r.get("source_class"), wb_source=(r.get("colour") or {}).get("wb_source"), note=r.get("note"), clipped_pct=(r.get("stats") or {}).get("clipped_pixels_pct")))


def worker():
    global studio
    try:
        import studio as engine
        studio = engine.Studio(status=lambda **k: set_state(**k)); set_state(ready=True, message="", device=studio.device, physics=dict(engine.PHYSICS_DEFAULTS))
    except Exception as e:
        traceback.print_exc(); set_state(error=f"Engine failed to start: {e}", message=""); return
    while True:
        kind, arg = jobs.get()
        set_state(busy=True, error=None)
        try:
            if kind == "open":
                if arg.get("purge"):  # leaving a photo: free its memory before loading the next one
                    studio.purge(); set_state(physics=dict(engine.PHYSICS_DEFAULTS))
                set_state(file=Path(arg["path"]).name, plan=None, capture=None, seconds={})
                s = studio.open(arg["path"], arg.get("physics") or state.get("physics"))
                set_state(physics=s.physics, plan=clean(s.plan), capture=capture_info(s), auto_ev=round(s.auto_ev, 2), exposure_info=clean(s.exp_rep), seconds=dict(s.seconds), generation=state["generation"] + 1)
            elif kind == "physics":
                s = studio.current
                if studio.input_key(s.physics) != studio.input_key(arg):
                    studio.open(str(s.path), arg); s = studio.current
                else:
                    studio.run_physics(s, arg)
                set_state(physics=s.physics, plan=clean(s.plan), auto_ev=round(s.auto_ev, 2), exposure_info=clean(s.exp_rep), seconds=dict(s.seconds), generation=state["generation"] + 1)
            elif kind == "export":
                rec = studio.export(arg["adjust"], arg["format"], OUT); s = studio.current
                set_state(exports=[clean(rec)] + state["exports"][:19], seconds=dict(s.seconds))
        except Exception as e:
            traceback.print_exc(); set_state(error=f"{type(e).__name__}: {str(e)[:300]}")
        set_state(busy=not jobs.empty(), engine=None, message="")  # still busy if another job is already waiting


def adjust_from(q):
    a = {}
    for k in ("oil", "oil_thick", "exposure", "temp", "tint", "contrast", "low", "mid", "high", "saturation", "vibrance"):
        a[k] = float(q.get(k, ["0"])[0])
    a["view"] = q.get("view", ["lightbox"])[0]
    return a


def file_entry(p, group):
    return dict(path=str(p), name=p.name, group=group, mb=round(p.stat().st_size / 1e6, 1))


def list_files():
    out = [file_entry(p, "Added") for p in extra_files if p.exists()]
    out += [file_entry(p, "Your photos") for p in sorted((ROOT / "samples").glob("*"), reverse=True) if p.suffix.lower() in EXT]
    t = ROOT / "samples" / "test_formats"
    for sub, g in (("raw", "Cameras and phones"), ("jxl", "JPEG XL")):
        out += [file_entry(p, g) for p in sorted((t / sub).glob("*")) if p.suffix.lower() in EXT]
    return out


def make_thumb(path):
    """Small embedded preview if the file has one, otherwise a decode. Returns JPEG bytes or None."""
    import cv2, numpy as np
    p = Path(path); key = (str(p), p.stat().st_mtime)
    if key in thumbs:
        return thumbs[key]
    im = None; suf = p.suffix.lower()
    try:
        if suf == ".jxl":
            import imagecodecs as ic
            a = ic.jpegxl_decode(p.read_bytes()); a = a if a.ndim == 3 else np.repeat(a[..., None], 3, -1)
            a = (a[..., :3] / (np.iinfo(a.dtype).max if a.dtype != np.float32 else 1) * 255).clip(0, 255).astype(np.uint8); im = a[..., ::-1].copy()
        elif suf in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
            im = cv2.imread(str(p))
        else:
            import rawpy
            with rawpy.imread(str(p)) as r:
                t = r.extract_thumb(); flip = r.sizes.flip
            im = cv2.imdecode(np.frombuffer(t.data, np.uint8), 1) if t.format == rawpy.ThumbFormat.JPEG else np.asarray(t.data)[..., ::-1].copy()
            if flip in (5, 6) and im.shape[0] < im.shape[1]:  # some previews are stored upright already
                im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE if flip == 5 else cv2.ROTATE_90_CLOCKWISE)
    except Exception:
        im = _tiff_preview(p)
    if im is None:
        thumbs[key] = None; return None
    k = 150 / im.shape[0]; ok, b = cv2.imencode(".jpg", cv2.resize(im, (max(1, int(im.shape[1] * k)), 150), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 78])
    thumbs[key] = b.tobytes() if ok else None
    return thumbs[key]


def _tiff_preview(p):
    import cv2, numpy as np, tifffile
    try:
        with tifffile.TiffFile(p) as t:
            page = t.pages[0]; sub = page.pages[0]; o = sub.tags["StripOffsets"].value; c = sub.tags["StripByteCounts"].value
            o = o[0] if isinstance(o, (tuple, list)) else o; c = c[0] if isinstance(c, (tuple, list)) else c
            t.filehandle.seek(o); im = cv2.imdecode(np.frombuffer(t.filehandle.read(c), np.uint8), 1); ori = page.tags["Orientation"].value if "Orientation" in page.tags else 1
        return im if int(ori) not in (6, 8) or im.shape[0] > im.shape[1] else cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE if int(ori) == 6 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        return None


def browse_dialog():
    """Native file picker, in its own process so Tk does not touch the server threads."""
    code = ("import tkinter as t, tkinter.filedialog as f; r=t.Tk(); r.withdraw(); r.attributes('-topmost',True);"
            "print(f.askopenfilename(title='Open a photo', filetypes=[('Photos', '" + " ".join("*" + e for e in sorted(EXT)) + "'), ('All files', '*.*')]))")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600).stdout.strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        self.send_response(code); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def _json(self, o, code=200):
        self._send(json.dumps(o, default=str).encode(), code=code)

    def _local_only(self):
        """Other web pages can send requests to localhost: only accept POSTs from our own page."""
        origin = self.headers.get("Origin")
        return origin is None or origin == f"http://{self.headers.get('Host')}"

    def _ready_image(self, q, fn):
        if studio is None or studio.current is None or studio.current.small_logH is None:
            return self._send(b"", "text/plain", 409)
        try:
            self._send(fn(adjust_from(q)), "image/jpeg")
        except Exception as e:
            if isinstance(e, ConnectionError):
                return
            traceback.print_exc(); self._send(str(e).encode(), "text/plain", 500)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            return self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/favicon.png":
            return self._send((ROOT / "assets" / "icon_256.png").read_bytes(), "image/png")
        if u.path == "/api/state":
            with lock:
                return self._json(state)
        if u.path == "/api/files":
            return self._json(list_files())
        if u.path == "/api/thumb":
            b = make_thumb(q["path"][0]) if Path(q["path"][0]).exists() else None
            return self._send(b, "image/jpeg") if b else self._send(b"", "text/plain", 404)
        if u.path == "/api/view.jpg":
            return self._ready_image(q, studio.view if studio else None)
        if u.path == "/api/before.jpg":
            return self._ready_image(q, studio.before if studio else None)
        if u.path == "/api/crop.jpg":
            return self._ready_image(q, lambda a: studio.crop(a, int(q["x"][0]), int(q["y"][0]), int(q["w"][0]), int(q["h"][0]), q.get("which", ["film"])[0]))
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if not self._local_only():
            return self._json(dict(error="forbidden"), 403)
        u = urlparse(self.path); n = int(self.headers.get("Content-Length", 0))
        if u.path == "/api/upload":  # drag and drop: copy into a temporary inbox
            INBOX.mkdir(exist_ok=True); name = Path(parse_qs(u.query)["name"][0]).name; p = INBOX / name
            with open(p, "wb") as f:
                left = n
                while left > 0:
                    c = self.rfile.read(min(1 << 20, left))
                    if not c:
                        break
                    f.write(c); left -= len(c)
            extra_files.insert(0, p); jobs.put(("open", dict(path=str(p), purge=True))); set_state(busy=True, error=None, file=p.name)
            return self._json(dict(ok=True, path=str(p)))
        d = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/browse":
            p = browse_dialog()
            if p:  # the page opens it, after asking about any unsaved edit
                extra_files.insert(0, Path(p))
            return self._json(dict(path=p))
        if u.path == "/api/open":
            if not Path(d["path"]).is_file():
                return self._json(dict(error="file not found"), 404)
            jobs.put(("open", dict(path=d["path"], physics=d.get("physics"), purge=d.get("purge", False)))); set_state(busy=True, error=None, file=Path(d["path"]).name)
            return self._json(dict(ok=True))
        if u.path == "/api/physics":
            if studio is None or studio.current is None:
                return self._json(dict(error="no photo open"), 409)
            jobs.put(("physics", d)); set_state(busy=True, error=None); return self._json(dict(ok=True))
        if u.path == "/api/export":
            if studio is None or studio.current is None:
                return self._json(dict(error="no photo open"), 409)
            jobs.put(("export", d)); set_state(busy=True, error=None); return self._json(dict(ok=True))
        if u.path == "/api/reveal":
            OUT.mkdir(parents=True, exist_ok=True); os.startfile(OUT); return self._json(dict(ok=True))
        self._json(dict(error="not found"), 404)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # the page cancels old previews while a slider moves: not an error
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def open_window(url):
    edge = shutil.which("msedge") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if os.path.exists(edge):
        subprocess.Popen([edge, f"--app={url}", "--window-size=1600,1000"])  # app window without browser chrome
    else:
        webbrowser.open(url)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--no-browser", action="store_true"); a = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    srv = Server(("127.0.0.1", a.port), Handler); url = f"http://127.0.0.1:{a.port}/"; print("ColorForge running at", url, flush=True)
    if not a.no_browser:
        open_window(url)
    srv.serve_forever()
