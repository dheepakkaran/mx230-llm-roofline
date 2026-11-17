"""Render film/keynote.html into the keynote film (1920x1080, 30 fps) with a synthesized score.

The page exposes renderAt(t), DURATION, HITS and PULSE. Frames are screenshotted with Playwright's
bundled Chromium in parallel workers, in chunks of 120 frames (each chunk is its own browser and its own
MP4, so a crashed run resumes where it stopped). The score is made here with numpy: a slow pad, a soft
plucked arpeggio, a pulse under the results and a low hit on each scene cut. No samples, nothing downloaded.

Steps (scratch files go to film/_work/, which is gitignored):
  PLAYWRIGHT_BROWSERS_PATH=~/workspace/.tools/ms-playwright ~/workspace/.tools/pw-venv/bin/python film/make_film.py frames
  pixi run python film/make_film.py score          # numpy
  python3 film/make_film.py mux                    # ffmpeg -> docs/media/keynote.mp4
  python3 film/make_film.py previews               # keynote.jpg poster + keynote.webp silent preview
  ... make_film.py stills 3 20 45 80 115           # PNG stills into film/_work/stills/
  FILM=upstream ... = the llama.cpp PR story film (film/upstream.html -> docs/media/upstream.*, scratch in film/_work/upstream/)
"""
import json
import os
import pathlib
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FILM = os.environ.get("FILM", "keynote")  # keynote | upstream (film/<FILM>.html -> docs/media/<FILM>.*)
WORK = HERE / "_work" if FILM == "keynote" else HERE / "_work" / FILM
CHUNKS = WORK / "chunks"
MEDIA = ROOT / "docs" / "media"
PAGE = (HERE / f"{FILM}.html").as_uri() + "?manual=1"
FPS, SR, CHUNK, WORKERS = 30, 44100, 120, 6
BEAT = 60 / 90


def open_page(p):
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(PAGE)
    page.evaluate("document.fonts.ready.then(() => 0)")
    page.wait_for_function("document.fonts.check('700 100px Inter')")
    return browser, page


def meta():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page = open_page(p)
        d = page.evaluate("({dur: DURATION, hits: HITS, pulse: PULSE})")
        browser.close()
    (WORK / "meta.json").write_text(json.dumps(d))
    return d


def render_chunk(args):
    k, f0, f1 = args
    out = CHUNKS / f"c{k:03d}.mp4"
    if out.exists():
        return out
    from playwright.sync_api import sync_playwright
    tmp = CHUNKS / f"c{k:03d}.part.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", str(tmp)],
                           stdin=subprocess.PIPE)
    with sync_playwright() as p:
        browser, page = open_page(p)
        for f in range(f0, f1):
            page.evaluate(f"renderAt({f / FPS})")
            enc.stdin.write(page.screenshot(type="png"))
        browser.close()
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on chunk {k}")
    tmp.rename(out)
    return out


def frames():
    CHUNKS.mkdir(parents=True, exist_ok=True)
    d = meta()
    total = int(round(d["dur"] * FPS))
    jobs = [(k, f0, min(total, f0 + CHUNK)) for k, f0 in enumerate(range(0, total, CHUNK))]
    with ProcessPoolExecutor(WORKERS) as ex:
        for out in ex.map(render_chunk, jobs):
            print("done", out.name, flush=True)
    (WORK / "chunks.txt").write_text("".join(f"file '{CHUNKS / f'c{k:03d}.mp4'}'\n" for k, _, _ in jobs))


def stills(times):
    from playwright.sync_api import sync_playwright
    (WORK / "stills").mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, page = open_page(p)
        for t in times:
            page.evaluate(f"renderAt({t})")
            page.screenshot(path=str(WORK / "stills" / f"t{float(t):06.1f}.png"))
        browser.close()


# ---------------------------------------------------------------- score
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def score():
    import numpy as np
    d = json.loads((WORK / "meta.json").read_text())
    dur, hits, pulse = d["dur"], d["hits"], d["pulse"]
    n = int(dur * SR)
    t = np.arange(n) / SR
    L, R = np.zeros(n), np.zeros(n)
    # Dm9 - Bbmaj7 - Fadd9 - C: an open, slightly hopeful loop, two bars per chord
    prog = [(38, [57, 60, 64, 65]), (34, [57, 62, 65, 69]), (41, [60, 65, 67, 69]), (36, [60, 64, 67, 72])]
    span = 8 * BEAT
    for k in range(int(dur / span) + 2):
        root, tones = prog[k % 4]
        c = k * span + span / 2
        env = np.clip(1 - np.abs((t - c) / (span / 2 + 1.0)), 0, 1)
        env = 0.5 - 0.5 * np.cos(np.pi * env)
        if not env.any():
            continue
        for m, g in [(root, 0.20), (root + 12, 0.08)] + [(x, 0.05) for x in tones]:
            f = midi(m)
            sl = slice(*np.flatnonzero(env)[[0, -1]] + [0, 1])
            tt = t[sl]
            L[sl] += g * env[sl] * (np.sin(2 * np.pi * f * 0.9985 * tt) + 0.12 * np.sin(4 * np.pi * f * tt))
            R[sl] += g * env[sl] * (np.sin(2 * np.pi * f * 1.0015 * tt + 0.4) + 0.12 * np.sin(4 * np.pi * f * tt + 0.2))
    # arpeggio: sixteenth-ish plucks, enters after the title, fuller from the chip scene on
    step = BEAT / 2
    pl = int(0.6 * SR)
    pt = np.arange(pl) / SR
    pattern = [0, 2, 1, 3, 2, 1, 3, 2]
    for i in range(int(dur / step)):
        s = i * step
        if s < 6.0 or s > dur - 4:
            continue
        lvl = 0.5 + 0.5 * min(1, max(0, (s - 15) / 6))
        _, tones = prog[int(s // span) % 4]
        f = midi(tones[pattern[i % 8]] + 12)
        note = (np.sin(2 * np.pi * f * pt) + 0.25 * np.sin(4 * np.pi * f * pt) * np.exp(-pt / 0.08)) * np.exp(-pt / 0.28)
        note *= np.clip(pt / 0.004, 0, 1)
        a = int(s * SR)
        b = min(n, a + pl)
        g = 0.055 * lvl
        pan = 0.5 + 0.35 * np.sin(i * 0.9)
        L[a:b] += g * (1 - pan) * 2 * note[: b - a]
        R[a:b] += g * pan * 2 * note[: b - a]
    # pulse: soft kick on each beat plus a quiet off-beat shaker while the results run
    kl = int(0.4 * SR)
    kt = np.arange(kl) / SR
    kick = np.sin(2 * np.pi * (48 * kt + 70 * 0.035 * (1 - np.exp(-kt / 0.035)))) * np.exp(-kt / 0.13)
    rng = np.random.default_rng(7)
    shl = int(0.06 * SR)
    shaker = rng.standard_normal(shl) * np.exp(-np.arange(shl) / SR / 0.012)
    shaker = np.diff(shaker, prepend=0)  # brighter
    for i in range(int(dur / BEAT)):
        s = i * BEAT
        if not (pulse[0] <= s < pulse[1]):
            continue
        ramp = min(1, (s - pulse[0]) / 4) * min(1, (pulse[1] - s) / 3)
        for src, off, g in [(kick, 0, 0.13 * ramp), (shaker, BEAT / 2, 0.025 * ramp)]:
            a = int((s + off) * SR)
            b = min(n, a + len(src))
            if a >= n:
                continue
            L[a:b] += g * src[: b - a]
            R[a:b] += g * src[: b - a] * (0.8 if off else 1)
    # hits on scene cuts: low boom + airy riser before it
    for s, g in hits:
        hl = int(2.8 * SR)
        ht = np.arange(hl) / SR
        boom = np.sin(2 * np.pi * (36 * ht + 44 * 0.07 * (1 - np.exp(-ht / 0.07)))) * np.exp(-ht / 0.7)
        air = rng.standard_normal(hl) * np.exp(-ht / 0.05) * 0.03
        a = int(s * SR)
        b = min(n, a + hl)
        L[a:b] += g * (boom + air)[: b - a]
        R[a:b] += g * (boom - air)[: b - a]
        wl = int(1.0 * SR)
        w = rng.standard_normal(wl) * np.linspace(0, 1, wl) ** 3 * 0.025 * g
        a0 = max(0, a - wl)
        L[a0:a] += w[wl - (a - a0):]
        R[a0:a] += w[wl - (a - a0):]
    # final chord swell under the outro
    end0 = dur - 9
    env = np.clip((t - end0) / 3, 0, 1) * np.clip((dur - t) / 4, 0, 1)
    for m in [50, 57, 62, 65, 69]:
        L += 0.05 * env * np.sin(2 * np.pi * midi(m) * t)
        R += 0.05 * env * np.sin(2 * np.pi * midi(m) * 1.002 * t + 0.5)
    # echo, fades, normalise
    dly = int(BEAT * 0.75 * SR)
    for ch in (L, R):
        ch[dly:] += 0.30 * ch[:-dly].copy()
        ch[2 * dly:] += 0.10 * ch[:-2 * dly].copy()
    fade = np.clip(t / 1.5, 0, 1) * np.clip((dur - t) / 3.5, 0, 1)
    mix = np.stack([L * fade, R * fade], axis=1)
    mix = (mix / np.abs(mix).max() * 0.82 * 32767).astype(np.int16)
    with wave.open(str(WORK / "score.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(mix.tobytes())
    print("score.wav", dur, "s")


def mux():
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / f"{FILM}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(WORK / "chunks.txt"),
                    "-i", str(WORK / "score.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", os.environ.get("CRF", "23"),
                    "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    print(out, round(out.stat().st_size / 1e6, 1), "MB")


def previews(poster_t="57.6"):
    src = MEDIA / f"{FILM}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", poster_t, "-i", str(src), "-frames:v", "1",
                    "-vf", "scale=1280:720:flags=lanczos", "-q:v", "3", str(MEDIA / f"{FILM}.jpg")], check=True)
    # silent preview at 4x speed, 960 px wide, 8 fps
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an",
                    "-vf", "setpts=PTS/4,fps=8,scale=960:540:flags=lanczos", "-c:v", "libwebp_anim", "-q:v", "60",
                    "-compression_level", "6", "-loop", "0", str(MEDIA / f"{FILM}.webp")], check=True)
    for f in (f"{FILM}.jpg", f"{FILM}.webp"):
        print(f, round((MEDIA / f).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "frames"
    if cmd == "frames":
        frames()
    elif cmd == "score":
        score()
    elif cmd == "mux":
        mux()
    elif cmd == "previews":
        previews(*sys.argv[2:3])
    elif cmd == "stills":
        stills(sys.argv[2:])
    elif cmd == "meta":
        print(meta())
