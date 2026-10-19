"""Robot-arm pick-and-place with JEV-9B-VL System 1 (POST /v1/decide), simulated in MuJoCo.

A gantry arm with a two-finger gripper above a table with three coloured cubes and a yellow tray. Every step the model
sees two camera images (front and top; the top view shows a black dot under the gripper) and the gripper state, and
picks one of 8 actions in one forward pass. Grasping is simplified: closing the gripper low over a cube attaches the
cube; opening releases it and it falls under MuJoCo physics.

    MUJOCO_GL=glfw DISPLAY=:99 python3 robot_arm.py run 20
    MUJOCO_GL=glfw DISPLAY=:99 python3 robot_arm.py record SEED OUT.mp4
"""
import base64, io, json, os, random, subprocess, sys, tempfile, time
os.environ.setdefault("MUJOCO_GL", "glfw"); os.environ.setdefault("DISPLAY", ":99")
import numpy as np, mujoco, requests
from PIL import Image, ImageDraw, ImageFont

API = os.environ.get("JEV_URL", "http://localhost:8000") + "/v1/decide"
HERE = os.path.dirname(os.path.abspath(__file__))
FONT, FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ACTIONS = ["move left", "move right", "move forward (away from the front camera)", "move backward (toward the front camera)",
           "lower the gripper", "raise the gripper", "close the gripper (grasp)", "open the gripper (release)"]
COLORS = {"red": "0.85 0.15 0.15 1", "green": "0.15 0.65 0.25 1", "blue": "0.15 0.3 0.85 1"}
STEP, ZSTEP, ZTOP, ZLOW, HALF = 0.04, 0.05, 0.25, 0.05, 0.02
MAX_STEPS = 45


def xml(cubes, tray):
    cube_xml = "".join(f'<body name="{n}" pos="{x} {y} {HALF + 0.001}"><freejoint name="j_{n}"/>'
                       f'<geom type="box" size="{HALF} {HALF} {HALF}" rgba="{COLORS[n]}" mass="0.05" friction="1 0.01 0.001"/></body>'
                       for n, (x, y) in cubes.items())
    tx, ty = tray
    return f"""<mujoco><option timestep="0.002"/>
<visual><headlight ambient=".3 .3 .3" diffuse=".3 .3 .3"/><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
<asset><texture name="g" type="2d" builtin="checker" rgb1=".72 .70 .66" rgb2=".62 .60 .56" width="256" height="256"/>
<material name="g" texture="g" texrepeat="6 6"/></asset>
<worldbody>
<light pos="0 0 1.6" dir="0 0 -1" diffuse=".45 .45 .45" castshadow="false"/>
<geom type="plane" size=".45 .45 .01" material="g"/>
<geom type="box" size=".012 .012 .22" pos="-.42 .42 .22" rgba=".35 .35 .4 1" group="2"/><geom type="box" size=".012 .012 .22" pos=".42 .42 .22" rgba=".35 .35 .4 1" group="2"/>
<geom type="box" size=".43 .012 .012" pos="0 .42 .44" rgba=".35 .35 .4 1" group="2"/>
<body name="tray" pos="{tx} {ty} 0">
  <geom type="box" size=".065 .065 .003" pos="0 0 .003" rgba=".95 .8 .1 1"/>
  <geom type="box" size=".065 .004 .02" pos="0 .061 .02" rgba=".95 .8 .1 1"/><geom type="box" size=".065 .004 .02" pos="0 -.061 .02" rgba=".95 .8 .1 1"/>
  <geom type="box" size=".004 .065 .02" pos=".061 0 .02" rgba=".95 .8 .1 1"/><geom type="box" size=".004 .065 .02" pos="-.061 0 .02" rgba=".95 .8 .1 1"/>
</body>
{cube_xml}
<body name="dot" mocap="true" pos="0 0 .0015"><geom type="cylinder" size=".012 .001" rgba="0 0 0 0.85" contype="0" conaffinity="0"/></body>
<body name="arm" mocap="true" pos="0 0 {ZTOP}">
  <geom type="cylinder" size=".012 .2" pos="0 0 .23" rgba=".55 .55 .6 1" contype="0" conaffinity="0" group="2"/>
  <geom type="box" size=".035 .02 .015" rgba=".25 .25 .3 1" contype="0" conaffinity="0" group="2"/>
  <camera name="wrist" pos="0 0 -0.02" xyaxes="1 0 0 0 1 0" fovy="70"/>
</body>
<body name="fl" mocap="true" pos="-.03 0 {ZTOP - .035}"><geom type="box" size=".005 .015 .03" rgba=".15 .15 .15 1" contype="0" conaffinity="0" group="2"/></body>
<body name="fr" mocap="true" pos=".03 0 {ZTOP - .035}"><geom type="box" size=".005 .015 .03" rgba=".15 .15 .15 1" contype="0" conaffinity="0" group="2"/></body>
<camera name="front" pos="0 -0.78 0.62" xyaxes="1 0 0 0 0.62 0.78"/>
<camera name="top" pos="0 0 1.05" xyaxes="1 0 0 0 1 0"/>
</worldbody></mujoco>"""


class Env:
    def __init__(self, seed):
        r = random.Random(seed)
        while True:  # cubes and tray apart from each other, inside reach
            pts = [(round(r.uniform(-.3, .3), 3), round(r.uniform(-.25, .25), 3)) for _ in range(4)]
            if all(np.hypot(a[0] - b[0], a[1] - b[1]) > .12 for i, a in enumerate(pts) for b in pts[i + 1:]):
                break
        self.cubes = dict(zip(COLORS, pts[:3])); self.tray = pts[3]
        self.target = r.choice(list(COLORS))
        self.m = mujoco.MjModel.from_xml_string(xml(self.cubes, self.tray)); self.d = mujoco.MjData(self.m)
        self.g = np.array([round(r.uniform(-.2, .2), 2), round(r.uniform(-.2, .2), 2), ZTOP]); self.closed = False; self.held = None
        self.ren = mujoco.Renderer(self.m, 300, 400); self.ren_top = mujoco.Renderer(self.m, 300, 300); self.ren_w = mujoco.Renderer(self.m, 300, 300)
        self._pose(); self.settle(200)

    def mocap(self, name):
        return self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)]

    def cube_q(self, n):
        return self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"j_{n}")]

    def cube_pos(self, n):
        a = self.cube_q(n); return self.d.qpos[a:a + 3].copy()

    def _pose(self):
        x, y, z = self.g; w = 0.022 if self.closed else 0.03
        self.d.mocap_pos[self.mocap("arm")] = [x, y, z]; self.d.mocap_pos[self.mocap("dot")] = [x, y, .0015]
        self.d.mocap_pos[self.mocap("fl")] = [x - w, y, z - .035]; self.d.mocap_pos[self.mocap("fr")] = [x + w, y, z - .035]
        if self.held:
            a = self.cube_q(self.held); self.d.qpos[a:a + 3] = [x, y, z - .045]; self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            va = self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"j_{self.held}")]; self.d.qvel[va:va + 6] = 0

    def settle(self, n=300):
        for _ in range(n):
            self._pose(); mujoco.mj_step(self.m, self.d)

    def step(self, a):
        x, y, z = self.g
        if a == 0: x -= STEP
        elif a == 1: x += STEP
        elif a == 2: y += STEP
        elif a == 3: y -= STEP
        elif a == 4: z = max(ZLOW, z - ZSTEP)
        elif a == 5: z = min(ZTOP, z + ZSTEP)
        elif a == 6:
            self.closed = True
            if self.held is None and z <= ZLOW + 1e-6:
                near = [n for n in COLORS if np.hypot(*(self.cube_pos(n)[:2] - [x, y])) < 0.03 and self.cube_pos(n)[2] < .06]
                self.held = near[0] if near else None
        elif a == 7:
            self.closed = False; self.held = None
        self.g = np.clip([x, y, z], [-.38, -.32, ZLOW], [.38, .36, ZTOP]); self.settle()

    def done(self):
        p = self.cube_pos(self.target)
        return self.held is None and abs(p[0] - self.tray[0]) < .06 and abs(p[1] - self.tray[1]) < .06 and p[2] < .06

    def wrist(self):
        opt = mujoco.MjvOption(); opt.geomgroup[2] = 0
        self.ren_w.update_scene(self.d, camera="wrist", scene_option=opt); im = Image.fromarray(self.ren_w.render())
        d = ImageDraw.Draw(im); c = im.width // 2
        d.line((c - 18, c, c + 18, c), fill=(255, 255, 255), width=2); d.line((c, c - 18, c, c + 18), fill=(255, 255, 255), width=2)
        return im

    def images(self):
        self.ren.update_scene(self.d, camera="front"); f = Image.fromarray(self.ren.render())
        opt = mujoco.MjvOption(); opt.geomgroup[2] = 0  # hide gantry and arm column from above
        self.ren_top.update_scene(self.d, camera="top", scene_option=opt); t = Image.fromarray(self.ren_top.render())
        return f, t


def url(im):
    b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def episode(seed, frames=None):
    env = Env(seed); hist, lat, ok = [], [], False
    for step in range(MAX_STEPS):
        f, t = env.images()
        grip = ("closed, holding the " + env.held + " cube") if env.held else ("closed, holding nothing" if env.closed else "open")
        state = ["Front camera:", {"image": url(f)}, "\nTop camera (the black dot is directly under the gripper; up in this image is forward):",
                 {"image": url(t)}, f"\nTask: pick up the {env.target} cube and put it in the yellow tray.\nGripper: {grip}.\n"
                 f"Gripper height: {'raised' if env.g[2] >= ZTOP - 1e-6 else ('lowest (at cube height)' if env.g[2] <= ZLOW + 1e-6 else 'in between')}.\n"
                 f"Last actions: {'; '.join(hist[-3:]) if hist else 'none'}"]
        t0 = time.time()
        r = requests.post(API, json={"kind": "choice", "state": state, "question": "Which action should the robot take next?",
                                     "options": ACTIONS, "thinking": "off"}, timeout=120).json()
        lat.append(time.time() - t0)
        p = r["probabilities"]; a = int(np.argmax(p))
        if frames is not None:
            frames.append({"front": f, "top": t, "probs": p, "a": a, "step": step + 1, "ms": 1000 * lat[-1], "task": env.target, "grip": grip})
        env.step(a); hist.append(ACTIONS[a].split(" (")[0])
        if env.done():
            ok = True
            if frames is not None:
                f, t = env.images(); frames.append({"front": f, "top": t, "probs": None, "a": None, "step": step + 1, "ms": 0, "task": env.target, "grip": "open"})
            break
    held_ever = any(h == "close the gripper" for h in hist)
    return {"seed": seed, "target": env.target, "success": ok, "steps": len(hist), "actions": hist, "latency_s": lat}


def run(n=20):
    res = [episode(s) for s in range(1, n + 1)]
    lat = sorted(x for r in res for x in r["latency_s"])
    summ = {"episodes": n, "success": sum(r["success"] for r in res) / n, "mean_steps_success": float(np.mean([r["steps"] for r in res if r["success"]] or [0])),
            "median_ms": 1000 * lat[len(lat) // 2]}
    json.dump({"summary": summ, "episodes": res}, open(os.path.join(HERE, "arm_results.json"), "w"), indent=1)
    print(json.dumps(summ)); [print(r["seed"], r["target"], r["success"], r["steps"], " ".join(a.split()[0][:2] + a.split()[-1][:2] for a in r["actions"])) for r in res]


def frame(fr, W=1280, H=720):
    can = Image.new("RGB", (W, H), (248, 248, 250)); d = ImageDraw.Draw(can)
    f, fb, fs = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONTB, 19), ImageFont.truetype(FONT, 15)
    d.text((14, 12), f"{os.environ.get('JEV_LABEL', 'JEV-9B')} System 1 · robot arm: camera images → action probabilities", fill=(20, 30, 50), font=fb)
    d.text((14, 44), f"Task: pick up the {fr['task']} cube and put it in the yellow tray   ·   gripper: {fr['grip']}", fill=(40, 40, 40), font=f)
    can.paste(fr["front"].resize((560, 420)), (14, 84)); can.paste(fr["top"].resize((420, 420)), (584, 84))
    d.text((14, 510), "front camera", fill=(90, 90, 90), font=fs); d.text((584, 510), "top camera", fill=(90, 90, 90), font=fs)
    x0 = 1018
    if fr["probs"] is None:
        d.text((x0, 100), "Task complete", fill=(0, 140, 60), font=fb)
    else:
        d.text((x0, 90), f"Step {fr['step']} · {fr['ms']:.0f} ms", fill=(20, 30, 50), font=fb)
        y = 130
        for i, a in enumerate(ACTIONS):
            p = fr["probs"][i]; c = (200, 0, 0) if i == fr["a"] else (120, 140, 170)
            d.text((x0, y), a.split(" (")[0], fill=(200, 0, 0) if i == fr["a"] else (30, 30, 30), font=fs)
            d.rectangle((x0, y + 21, x0 + 240, y + 31), fill=(225, 228, 235)); d.rectangle((x0, y + 21, x0 + int(240 * p), y + 31), fill=c)
            d.text((x0 + 200, y), f"{p:.2f}", fill=(30, 30, 30), font=fs); y += 46
    return can


def record(seed, out):
    frames = []; res = episode(seed, frames); tmp = tempfile.mkdtemp(); n = 0
    for fr in frames:
        im = frame(fr)
        for _ in range(2 if fr["probs"] is not None else 6):
            im.save(f"{tmp}/{n:04d}.png"); n += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "fps=12", out], check=True)
    print(out, "success" if res["success"] else "FAILED", res["steps"], "steps")


if __name__ == "__main__" and sys.argv[1] in ("run", "view", "record"):
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif sys.argv[1] == "view":
        e = Env(int(sys.argv[2])); f, t = e.images(); f.save("/tmp/opencode/arm_front.png"); t.save("/tmp/opencode/arm_top.png"); print(e.target, e.cubes, e.tray, e.g)
    else:
        record(int(sys.argv[2]), sys.argv[3])


# ---------------------------------------------------------------- v2: System 1 visual servoing (binary decisions)
ZCARRY = 0.15


def ask(imgs, q, opts):
    state = []
    for title, im in imgs:
        state += [title, {"image": url(im)}, "\n"]
    t0 = time.time()
    p = requests.post(API, json={"kind": "choice", "state": state, "question": q, "options": opts, "thinking": "off"},
                      timeout=120).json()["probabilities"]
    return p, time.time() - t0


def zoom(env, t, half=60):
    """Crop of the top image centred on the black dot (projected position), upscaled 2.5x."""
    f = (t.height / 2) / np.tan(np.radians(45 / 2)); u = t.width / 2 + f * env.g[0] / 1.05; v = t.height / 2 - f * env.g[1] / 1.05
    u, v = int(np.clip(u, half, t.width - half)), int(np.clip(v, half, t.height - half))
    return t.crop((u - half, v - half, u + half, v + half)).resize((300, 300), Image.BICUBIC)


def servo(env, what, ref, log, frames, phase, max_iter=36):
    """Move the gripper over `what` by an adaptive bisection on each axis, driven only by two System 1 decisions per
    step: is `what` left or right of `ref`, and above or below it, in the top camera image (plus a zoomed view of the
    area around the gripper once the steps are small). A flipped answer halves that axis' step; three equal answers in a
    row double it."""
    step, hist = {"x": 0.08, "y": 0.08}, {"x": [], "y": []}
    for it in range(max_iter):
        f, t = env.images()
        imgs = [("Top camera image:", t)]
        if max(step.values()) <= 0.04:
            imgs.append(("Zoomed top view around the black dot (2.5x):", zoom(env, t)))
        plr, l1 = ask(imgs, f"In the top camera view, is the {what} to the left or to the right of the {ref}?", ["left", "right"])
        pud, l2 = ask(imgs, f"In the top camera view, is the {what} above or below the {ref}?", ["above", "below"])
        sx = -1 if plr[0] > plr[1] else 1; sy = 1 if pud[0] > pud[1] else -1
        for ax, sgn in (("x", sx), ("y", sy)):
            h = hist[ax]
            if h and sgn != h[-1]:
                step[ax] /= 2
            elif len(h) >= 2 and h[-1] == h[-2] == sgn:
                step[ax] = min(0.08, step[ax] * 2)
            h.append(sgn)
        log["decisions"] += 2; log["lat"] += [l1, l2]
        if frames is not None:
            frames.append({"front": f, "top": t, "phase": phase, "plr": plr, "pud": pud, "ms": 1000 * (l1 + l2), "task": env.target,
                           "grip": "holding the " + env.held + " cube" if env.held else "open"})
        if step["x"] < 0.012 and step["y"] < 0.012:
            return
        env.g = np.clip(env.g + [sx * step["x"], sy * step["y"], 0], [-.38, -.32, ZLOW], [.38, .36, ZTOP]); env.settle(150)


def episode2(seed, frames=None):
    env = Env(seed); log = {"decisions": 0, "lat": []}
    servo(env, f"{env.target} cube", "black dot", log, frames, "1 · move over the cube")
    c = env.cube_pos(env.target); log["err_cube_cm"] = round(100 * float(np.hypot(c[0] - env.g[0], c[1] - env.g[1])), 1)
    env.g[2] = ZLOW; env.settle(); env.step(6)  # lower and grasp
    grasped = env.held == env.target
    env.g[2] = ZCARRY; env.settle()
    servo(env, "yellow tray", f"{env.target} cube held by the gripper", log, frames, "2 · carry it over the tray")
    log["err_tray_cm"] = round(100 * float(np.hypot(env.tray[0] - env.g[0], env.tray[1] - env.g[1])), 1)
    env.step(7); env.settle(600)  # release
    ok = env.done()
    if frames is not None:
        f, t = env.images(); frames.append({"front": f, "top": t, "phase": "done" if ok else "missed", "plr": None, "pud": None, "ms": 0,
                                            "task": env.target, "grip": "open"})
    return {"seed": seed, "target": env.target, "grasped": bool(grasped), "success": bool(ok), "decisions": log["decisions"], "latency_s": log["lat"],
            "err_cube_cm": log.get("err_cube_cm"), "err_tray_cm": log.get("err_tray_cm")}


def run2(n=20):
    t0 = time.time(); res = [episode2(s) for s in range(1, n + 1)]
    lat = sorted(x for r in res for x in r["latency_s"])
    summ = {"episodes": n, "success": sum(r["success"] for r in res) / n, "grasped": sum(r["grasped"] for r in res) / n,
            "mean_decisions": float(np.mean([r["decisions"] for r in res])), "median_ms": 1000 * lat[len(lat) // 2], "seconds": time.time() - t0}
    json.dump({"summary": summ, "episodes": res}, open(os.path.join(HERE, "arm2_results.json"), "w"), indent=1)
    print(json.dumps(summ)); [print(r["seed"], r["target"], "grasp", r["grasped"], "success", r["success"], r["decisions"], "err cube/tray cm", r["err_cube_cm"], r["err_tray_cm"]) for r in res]


def frame2(fr, W=1280, H=720):
    can = Image.new("RGB", (W, H), (248, 248, 250)); d = ImageDraw.Draw(can)
    f, fb, fs = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONTB, 19), ImageFont.truetype(FONT, 16)
    d.text((14, 12), f"{os.environ.get('JEV_LABEL', 'JEV-9B')} System 1 · robot arm: camera image → two yes/no-style decisions per step", fill=(20, 30, 50), font=fb)
    d.text((14, 44), f"Task: pick up the {fr['task']} cube and put it in the yellow tray   ·   gripper: {fr['grip']}", fill=(40, 40, 40), font=f)
    can.paste(fr["front"].resize((560, 420)), (14, 84)); can.paste(fr["top"].resize((420, 420)), (584, 84))
    d.text((14, 510), "front camera", fill=(90, 90, 90), font=fs); d.text((584, 510), "top camera (what System 1 reads)", fill=(90, 90, 90), font=fs)
    x0 = 1016; d.text((x0, 90), fr["phase"].replace(" · ", ". "), fill=(20, 30, 50), font=ImageFont.truetype(FONTB, 16))
    if fr["plr"] is None:
        d.text((x0, 130), "Task complete" if fr["phase"] == "done" else "Missed", fill=(0, 140, 60) if fr["phase"] == "done" else (200, 0, 0), font=fb)
    else:
        d.text((x0, 125), f"{fr['ms']:.0f} ms for 2 decisions", fill=(60, 60, 60), font=fs)
        y = 165
        for title, labs, p in (("target left or right?", ["left", "right"], fr["plr"]), ("target above or below?", ["above", "below"], fr["pud"])):
            d.text((x0, y), title, fill=(30, 30, 30), font=fs); y += 26
            for lab, v in zip(labs, p):
                c = (200, 0, 0) if v == max(p) else (120, 140, 170)
                d.text((x0, y), lab, fill=(30, 30, 30), font=fs); d.rectangle((x0 + 62, y + 4, x0 + 62 + int(130 * v), y + 16), fill=c)
                d.text((x0 + 200, y), f"{v:.2f}", fill=(30, 30, 30), font=fs); y += 26
            y += 18
        d.text((x0, y + 10), "step size halves each time\na decision flips (bisection)", fill=(90, 90, 90), font=fs)
    return can


def record2(seed, out):
    frames = []; res = episode2(seed, frames); tmp = tempfile.mkdtemp(); n = 0
    for fr in frames:
        im = frame2(fr)
        for _ in range(2 if fr["plr"] is not None else 8):
            im.save(f"{tmp}/{n:04d}.png"); n += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "fps=12", out], check=True)
    print(out, "success" if res["success"] else "FAILED", res["decisions"], "decisions")


if __name__ == "__main__" and sys.argv[1] in ("run2", "record2"):
    run2(int(sys.argv[2])) if sys.argv[1] == "run2" else record2(int(sys.argv[2]), sys.argv[3])
