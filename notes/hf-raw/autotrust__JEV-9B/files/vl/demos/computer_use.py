"""Computer-use click loop with JEV-9B-VL System 1 (POST /v1/decide), in a real headless Chromium (Playwright).

Every step: screenshot -> every interactive element gets a numbered red box (Set-of-Mark) -> one System 1 `choice`
over "click element k" for each mark plus "the task is complete" -> the browser really clicks the chosen element.
Variants: "marks" (the options are only the mark numbers: the model must read the screenshot to know what each mark
is) and "marks+text" (each option also carries the element's visible text, like an accessibility tree).
Tasks are randomised per seed (which email / product / colour / size / setting; element order and contents shuffled).

    python3 computer_use.py run          # 3 apps x N seeds x 2 variants -> cu_results.json
    python3 computer_use.py record APP SEED VARIANT OUT.mp4
"""
import base64, io, json, os, sys, time, subprocess, tempfile
import requests
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "file://" + os.path.join(HERE, "webapp.html")
API = os.environ.get("JEV_URL", "http://localhost:8000") + "/v1/decide"
VW, VH = 1100, 640
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MAX_STEPS = 10


def task_for(app, world, seed):
    import random
    r = random.Random(seed * 7919 + len(app))
    if app == "mail":
        m = r.choice(world["MAILS"])
        return f"Archive the email from {m['from']} about the {m['subject']}.", {"mail": m["id"]}
    if app == "shop":
        p, c, s = r.choice(world["PRODUCTS"]), r.choice(world["COLORS"]), r.choice(world["SIZES"])
        return f"Buy a {c} {p['name']} in size {s}: add it to the cart and check out.", {"name": p["name"], "color": c, "size": s}
    sec = r.choice(list(world["SECTIONS"])); t = r.choice(world["SECTIONS"][sec])
    return f"Turn on \"{t}\" in the {sec} settings and save the change.", {"toggle": t}


def goal_met(app, S, g):
    if app == "mail":
        return S["archived"] == [g["mail"]] and not S["deleted"]
    if app == "shop":
        return bool(S["ordered"]) and S["cart"] == [{"name": g["name"], "color": g["color"], "size": g["size"]}]
    on = [t for t, v in (S["saved"] or {}).items() if v]
    return on == [g["toggle"]]


ELEMS_JS = """() => [...document.querySelectorAll('[data-click]')].map(e => { const r = e.getBoundingClientRect();
  return {x: r.x, y: r.y, w: r.width, h: r.height, text: (e.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 60),
          kind: e.classList.contains('swatch') ? 'colour swatch' : e.classList.contains('sw') ? 'switch' : e.classList.contains('row') ? 'list row' : e.classList.contains('card') ? 'product card' : 'button'}; })
  .filter(e => e.w > 0 && e.h > 0 && e.y < innerHeight)"""


def mark(png, elems):
    im = Image.open(io.BytesIO(png)).convert("RGB"); d = ImageDraw.Draw(im); f = ImageFont.truetype(FONTB, 15)
    for i, e in enumerate(elems):
        d.rectangle((e["x"] - 2, e["y"] - 2, e["x"] + e["w"] + 2, e["y"] + e["h"] + 2), outline=(230, 0, 0), width=2)
        lab = str(i + 1); tw = d.textlength(lab, font=f)
        ly = e["y"] - 20 if e["y"] >= 22 else e["y"] + e["h"] + 2  # label above the box, or below it at the top edge
        d.rectangle((e["x"] - 2, ly, e["x"] + tw + 6, ly + 18), fill=(230, 0, 0))
        d.text((e["x"] + 2, ly + 1), lab, fill="white", font=f)
    return im


def to_url(im):
    b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def episode(page, app, seed, variant, frames=None):
    page.goto(f"{URL}?app={app}&seed={seed}"); page.wait_for_timeout(50)
    world = page.evaluate("() => window.WORLD")
    task, goal = task_for(app, world, seed)
    hist, steps, stopped, lat = [], [], False, []
    for step in range(MAX_STEPS):
        elems = page.evaluate(ELEMS_JS)
        im = mark(page.screenshot(), elems)
        if variant == "marks":
            opts = [f"click element {i + 1}" for i in range(len(elems))]
        else:
            opts = [f"click element {i + 1}: {e['text'] or e['kind']}" + (f" ({e['kind']})" if e["text"] else "") for i, e in enumerate(elems)]
        opts.append("the task is complete")
        state = ["Browser screenshot. Every clickable element has a numbered red box.\n", {"image": to_url(im)},
                 f"\nTask: {task}\nActions taken so far: {'; '.join(hist) if hist else 'none'}"]
        t0 = time.time()
        r = requests.post(API, json={"kind": "choice", "state": state, "question": "What should be done next to complete the task?",
                                     "options": opts, "thinking": "off"}, timeout=120).json()
        lat.append(time.time() - t0)
        probs = r["probabilities"]; k = max(range(len(probs)), key=probs.__getitem__)
        if frames is not None:
            frames.append({"img": im, "task": task, "step": step + 1, "opts": opts, "probs": probs, "k": k, "ms": 1000 * lat[-1], "variant": variant})
        steps.append({"options": len(opts), "choice": opts[k], "p": probs[k]})
        if k == len(elems):
            stopped = True
            break
        e = elems[k]
        hist.append(f"clicked element {k + 1}" + (f" ({e['text'] or e['kind']})" if variant != "marks" else ""))
        page.mouse.click(e["x"] + e["w"] / 2, e["y"] + e["h"] / 2); page.wait_for_timeout(30)
    S = page.evaluate("() => window.S")
    ok = goal_met(app, S, goal)
    return {"app": app, "seed": seed, "variant": variant, "task": task, "success": bool(ok and stopped), "goal_met": bool(ok),
            "stopped": stopped, "steps": len(steps), "trace": steps, "latency_s": lat}


def worker(jobs):
    from playwright.sync_api import sync_playwright
    out = []
    with sync_playwright() as p:
        b = p.chromium.launch(); page = b.new_page(viewport={"width": VW, "height": VH})
        for app, seed, variant in jobs:
            out.append(episode(page, app, seed, variant))
        b.close()
    return out


def run(n=10, procs=6):
    from multiprocessing import Pool
    jobs = [(a, s, v) for v in ("marks", "marks+text") for a in ("mail", "shop", "settings") for s in range(1, n + 1)]
    chunks = [jobs[i::procs] for i in range(procs)]
    t0 = time.time()
    with Pool(procs) as pool:
        res = [r for part in pool.map(worker, chunks) for r in part]
    summ = {}
    for v in ("marks", "marks+text"):
        for a in ("mail", "shop", "settings", "all"):
            rs = [r for r in res if r["variant"] == v and (a == "all" or r["app"] == a)]
            lat = sorted(x for r in rs for x in r["latency_s"])
            summ[f"{v}/{a}"] = {"episodes": len(rs), "success": sum(r["success"] for r in rs) / len(rs),
                                "mean_steps": sum(r["steps"] for r in rs) / len(rs), "median_ms": 1000 * lat[len(lat) // 2]}
    json.dump({"summary": summ, "seconds": time.time() - t0, "episodes": res}, open(os.path.join(HERE, "cu_results.json"), "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:22s} success {v['success']:.0%}  steps {v['mean_steps']:.1f}  median {v['median_ms']:.0f} ms/decision (n={v['episodes']})")


def panel(fr, W=1280, H=720):
    """Video frame: marked screenshot on the left, task + option probabilities on the right."""
    can = Image.new("RGB", (W, H), (250, 250, 252)); d = ImageDraw.Draw(can)
    f, fb, fs = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONTB, 19), ImageFont.truetype(FONT, 15)
    shot = fr["img"].resize((880, int(880 * VH / VW))); can.paste(shot, (10, 80))
    d.text((14, 12), f"{os.environ.get('JEV_LABEL', 'JEV-9B')} System 1 · computer use: screenshot → which element to click", fill=(20, 30, 50), font=fb)
    d.text((14, 44), f"Task: {fr['task']}", fill=(40, 40, 40), font=f)
    x0 = 905; d.text((x0, 86), f"Step {fr['step']} · one forward pass · {fr['ms']:.0f} ms", fill=(20, 30, 50), font=fb)
    order = sorted(range(len(fr["probs"])), key=lambda i: -fr["probs"][i])[:6]
    y = 126
    for i in order:
        p = fr["probs"][i]; lab = fr["opts"][i].replace("click element", "click").replace("the task is complete", "task complete")
        lab = lab if len(lab) <= 34 else lab[:33] + "…"
        d.text((x0, y), lab, fill=(30, 30, 30) if i != fr["k"] else (200, 0, 0), font=fs)
        d.rectangle((x0, y + 21, x0 + 340, y + 33), fill=(225, 228, 235))
        d.rectangle((x0, y + 21, x0 + int(340 * p), y + 33), fill=(200, 0, 0) if i == fr["k"] else (120, 140, 170))
        d.text((x0 + 300, y), f"{p:.2f}", fill=(30, 30, 30), font=fs)
        y += 50
    note = ("options = numbered marks only\n(the model must read the screenshot)" if fr.get("variant") == "marks" else
            "options = mark number + visible text;\ncolour swatches and switches have no\ntext: the screenshot decides")
    d.text((x0, H - 80), note, fill=(90, 90, 90), font=fs)
    return can


def record(app, seed, variant, out):
    from playwright.sync_api import sync_playwright
    frames = []
    with sync_playwright() as p:
        b = p.chromium.launch(); page = b.new_page(viewport={"width": VW, "height": VH})
        res = episode(page, app, seed, variant, frames); b.close()
    tmp = tempfile.mkdtemp()
    n = 0
    for fr in frames:
        im = panel(fr)
        for _ in range(3):  # 1.5 s per decision at 2 fps
            im.save(f"{tmp}/{n:04d}.png"); n += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "2", "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "fps=10", out], check=True)
    print(out, "success" if res["success"] else "FAILED", res["steps"], "steps", [round(1000 * x) for x in res["latency_s"]], "ms")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        record(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
