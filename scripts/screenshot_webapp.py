"""Annotated screenshots of the web app for TOOLS.md.

    python scripts/screenshot_webapp.py        -> results/tools/webapp_*.png

Starts webapp/server.py on a spare port, drives headless Chrome over the
DevTools protocol (needs Chrome and the websocket-client package), and numbers
the parts of the page that TOOLS.md explains. Docker is not needed: the result
panel shows a stored run from results/runs/, as the Runs tab does.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tools"
APP_PORT, CDP_PORT = 8765, 9333
W, H, SCALE = 1600, 980, 1.5
RUN = "alternating_long_1029_150259"            # a stored run with a GIF and a summary
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
          "google-chrome", "chromium", "chromium-browser"]
BADGE, BOX = (227, 73, 72), (227, 73, 72, 255)


class Page:
    def __init__(self, ws_url):
        self.ws, self.n = websocket.create_connection(ws_url, timeout=60), 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps(dict(id=self.n, method=method, params=params)))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"{expr[:60]}...: {r['exceptionDetails']}")
        return r["result"].get("value")

    def wait(self, expr, timeout=30):
        t0 = time.time()
        while not self.js(f"!!({expr})"):            # a DOM node comes back as {}
            if time.time() - t0 > timeout:
                raise TimeoutError(expr)
            time.sleep(0.2)

    def rect(self, selector):
        return self.js(f"""(() => {{ const r = document.querySelector({json.dumps(selector)})
            .getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; }})()""")

    def shot(self, path):
        import base64
        data = self.call("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))


def font(size):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(path, marks):
    """marks: (number, [l, t, r, b] in CSS px, badge at 'tl', 'tr' or 'left' of the box)."""
    im = Image.open(path).convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f, rad = font(int(15 * SCALE)), int(13 * SCALE)
    for num, (l, t, r, b), corner in marks:
        box = [v * SCALE for v in (l - 3, t - 3, r + 3, b + 3)]
        d.rounded_rectangle(box, radius=int(8 * SCALE), outline=BOX, width=int(2 * SCALE))
        if corner == "left":                      # outside the box, for boxes full of text
            cx, cy = box[0] - rad - 2 * SCALE, (box[1] + box[3]) / 2
        else:
            cx = box[0] + rad * 0.6 if corner == "tl" else box[2] - rad * 0.6
            cy = box[1] + rad * 0.6
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=BADGE + (255,),
                  outline=(255, 255, 255, 255), width=int(2 * SCALE))
        d.text((cx, cy), str(num), fill="white", font=f, anchor="mm")
    Image.alpha_composite(im, over).convert("RGB").save(path, optimize=True)
    print("wrote", path)


def last_frame(run):
    import base64
    import io
    gif = Image.open(ROOT / "results" / "runs" / run / "deformation.gif")
    gif.seek(gif.n_frames - 1)
    buf = io.BytesIO()
    gif.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def find_chrome():
    for c in CHROME:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("Chrome or Edge not found")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, str(ROOT / "webapp" / "server.py"),
                               "--port", str(APP_PORT)], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    profile = tempfile.mkdtemp(prefix="kiri-chrome-")
    chrome = subprocess.Popen([find_chrome(), "--headless=new", f"--remote-debugging-port={CDP_PORT}",
                               f"--user-data-dir={profile}", "--hide-scrollbars", "--no-first-run",
                               "--remote-allow-origins=*", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json"))
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        p = Page(ws)
        p.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=SCALE,
               mobile=False)
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{APP_PORT}/api/generators")
                break
            except Exception:
                time.sleep(0.2)
        p.call("Page.navigate", url=f"http://127.0.0.1:{APP_PORT}/")
        p.wait("document.querySelectorAll('#samples .card').length > 0")

        # 1. a sample from the gallery, its generator form, validation, and a stored result
        p.js("""[...document.querySelectorAll('#samples .card')]
                .find(c => c.querySelector('.t').textContent === 'alternating long').click()""")
        p.wait("document.querySelector('#msgs').textContent.length > 0")
        p.js(f"""(async () => {{ const runs = await (await fetch('/api/runs')).json();
                 const r = runs.find(x => x.name === {json.dumps(RUN)});
                 showResult({{ ...r.summary, gif: r.gif, analysis: {{}} }}); }})()""")
        p.wait("document.querySelector('#result img') && document.querySelector('#result img').complete")
        # a screenshot catches whichever GIF frame is showing; show the last (most deformed) one
        p.js(f"document.querySelector('#result img').src = 'data:image/png;base64,{last_frame(RUN)}'")
        time.sleep(0.3)
        p.js("document.querySelector('#right').scrollTop = 0")
        time.sleep(0.5)
        marks = [(1, p.rect("#samples .card.active"), "tr"),
                 (2, p.rect("#genBox"), "tr"),
                 (3, p.rect("#canvasWrap"), "tr"),
                 (4, p.js("""(() => { const a = document.querySelector('#name').closest('.grid2')
                     .getBoundingClientRect(); return [a.left, a.top, a.right, a.bottom]; })()"""),
                  "tr"),
                 (5, p.rect("#msgs"), "left"),
                 (6, p.js("""(() => { const a = document.querySelector('#btnBuckle').parentElement
                     .getBoundingClientRect(); return [a.left, a.top, a.right, a.bottom]; })()"""),
                  "tr"),
                 (7, p.rect("#result"), "tr")]
        path = OUT / "webapp_sample.png"
        p.shot(path)
        annotate(path, marks)

        # 2. drawing your own: shapes, a selected shape's numbers, and a validation error
        p.js("""[...document.querySelectorAll('#samples .card')]
                .find(c => c.querySelector('.t').textContent === 'custom example').click()""")
        p.wait("shapes.length > 0")
        p.js("document.querySelector('#result').innerHTML = ''")
        # turn one slit so it crosses an edge crack: the live check reports it
        p.js("""(() => { const i = shapes.findIndex(s => s.type === 'slit');
                 shapes[i].p = [40, 45]; shapes[i].q = [60, 45];
                 selected = i; redraw(); renderProps(); scheduleValidate(); })()""")
        p.wait("document.querySelector('#props').children.length > 0")
        p.wait("document.querySelector('#msgs .err')")
        time.sleep(1.0)
        marks = [(1, p.js("""(() => { const a = document.querySelector('[data-tool]').parentElement
                     .getBoundingClientRect(); return [a.left, a.top, a.right, a.bottom]; })()"""),
                  "tr"),
                 (2, p.rect("#canvasWrap"), "tr"),
                 (3, p.rect("#props"), "tr"),
                 (4, p.rect("#msgs"), "left")]
        path = OUT / "webapp_draw.png"
        p.shot(path)
        annotate(path, marks)
    finally:
        chrome.terminate()
        server.terminate()
        time.sleep(0.5)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
