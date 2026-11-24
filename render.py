import glob, os, random, re, subprocess, textwrap
import numpy as np
import librosa

TOTAL = float(os.environ.get("DURATION") or 28)
MOOD = (os.environ.get("MOOD") or "hype").lower()
MOOD = MOOD if MOOD in ("hype", "emotional") else "hype"
OVERRIDE = (os.environ.get("CAPTION") or "").strip()
INTRO, PRE = 4.5, 1.2
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(FONT):
    FONT = None
os.makedirs("out", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)

def read_list(path):
    if not os.path.exists(path):
        return []
    return [l.strip().lower() for l in open(path) if l.strip() and not l.startswith("#")]

banned, favs = read_list("banned.txt"), read_list("favorites.txt")
CLIPS = [c for c in sorted(glob.glob("clips/*.mp4"))
         if not any(b in os.path.basename(c).lower() for b in banned)]
SONGS = glob.glob("songs/*.mp3")
if not CLIPS or not SONGS:
    raise SystemExit("need clips and songs")

# ---------- song choice: darker / minor key, energy by mood ----------
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def darkness(y, sr):
    mid = y[int(20 * sr):int(80 * sr)] if len(y) > 80 * sr else y
    ch = librosa.feature.chroma_stft(y=mid, sr=sr).mean(axis=1)
    bmaj = max(np.corrcoef(ch, np.roll(MAJ, k))[0, 1] for k in range(12))
    bmin = max(np.corrcoef(ch, np.roll(MIN, k))[0, 1] for k in range(12))
    cent = float(librosa.feature.spectral_centroid(y=mid, sr=sr).mean())
    return (1.0 if bmin > bmaj else 0.0) + (0.5 if cent < 2500 else 0.0)

loaded = []
for sp in random.sample(SONGS, min(6, len(SONGS))):
    y_, sr_ = librosa.load(sp, duration=150)
    loaded.append((sp, y_, sr_, darkness(y_, sr_), float(np.sqrt(np.mean(y_ ** 2)))))
emax = max(l[4] for l in loaded) or 1.0
def song_score(l):
    en = l[4] / emax
    return l[3] + (0.5 * en if MOOD == "hype" else 0.5 * (1 - en))
song, y, sr, dk, _ = max(loaded, key=song_score)
print("song:", song, "| darkness", dk)

dur_song = len(y) / sr
SONGLEN = min(TOTAL - INTRO, dur_song - PRE - 1)
if SONGLEN < 8:
    raise SystemExit("song too short")
TOTALV = INTRO + SONGLEN
rms = librosa.feature.rms(y=y, hop_length=sr)[0]
win = int(SONGLEN)
cands = [(float(rms[i:i + win].mean()), i) for i in range(int(PRE), len(rms) - win)]
offset = float(max(cands)[1]) if cands else PRE
seg_y = y[int(offset * sr):int((offset + SONGLEN) * sr)]
env = librosa.onset.onset_strength(y=seg_y, sr=sr)
_, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
bt = [float(t) for t in librosa.frames_to_time(beats, sr=sr) if t >= 0.3]

# ---------- pacing plan ----------
def beats_per_cut(p):
    if p < 0.25:
        return 2
    if p < 0.9:
        return 2 if random.random() < 0.13 else 1
    return 2

segs, prev, k, hold_done = [], 0.0, 0, False
while True:
    p = prev / SONGLEN
    slow = False
    if p >= 0.55 and not hold_done:
        n, slow, hold_done = 4, True, True
    else:
        n = beats_per_cut(p)
    idx = k + n - 1
    if idx >= len(bt):
        break
    t = bt[idx]
    if t >= SONGLEN - 0.3:
        break
    if t - prev < 0.25:
        k += 1
        continue
    segs.append({"a": prev, "b": t, "slow": slow})
    prev, k = t, idx + 1
segs.append({"a": prev, "b": SONGLEN, "slow": False})
print("segments:", len(segs))

# ---------- clip analysis ----------
def analyze(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf",
           "fps=4,scale=160:-2,signalstats,metadata=mode=print:file=-",
           "-an", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    rows, cur = [], {}
    for line in out.splitlines():
        if line.startswith("frame:"):
            if cur:
                rows.append(cur)
            cur = {}
        elif "lavfi.signalstats." in line:
            key, val = line.split("=", 1)
            try:
                cur[key.split(".")[-1]] = float(val)
            except ValueError:
                pass
    if cur:
        rows.append(cur)
    return rows

def sample_scores(rows):
    s, cut = [], []
    for r in rows:
        m = min(r.get("YDIF", 0) / 25, 1)
        con = (r.get("YHIGH", 200) - r.get("YLOW", 16)) / 255
        sat = min(r.get("SATAVG", 30) / 60, 1)
        yv = r.get("YAVG", 100)
        sc = (m + 0.5 * con + 0.3 * sat) if MOOD == "hype" else (0.3 * m + 0.9 * con + 0.3 * sat)
        if yv < 45:
            sc -= 0.6
        if yv > 200:
            sc -= 0.3
        s.append(sc)
        cut.append(1 if r.get("YDIF", 0) > 45 else 0)
    return np.array(s), np.array(cut)

data = {}
for c in CLIPS[:45]:
    rows = analyze(c)
    if len(rows) < 8:
        continue
    s, cut = sample_scores(rows)
    data[c] = (s, np.concatenate([[0], np.cumsum(s)]), np.concatenate([[0], np.cumsum(cut)]))
if not data:
    raise SystemExit("no usable clips")
print("analyzed clips:", len(data))

def candidates(L, recent):
    w = max(1, int(round(L * 4)))
    out = []
    for path, (s, cs, cc) in data.items():
        if path in recent or len(s) < w + 1:
            continue
        bonus = 0.3 if any(f in os.path.basename(path).lower() for f in favs) else 0.0
        for i in range(0, len(s) - w, 2):
            sc = (cs[i + w] - cs[i]) / w + bonus
            if cc[i + w] - cc[i + 1] > 0:
                sc -= 1.0
            out.append((float(sc), path, i * 0.25))
    out.sort(reverse=True)
    return out

def pick(L, recent, used):
    allc = candidates(L, recent) or candidates(L, set())
    good = []
    for sc, p, st in allc:
        if any(p == up and abs(st - ust) < L + 0.5 for up, ust in used):
            continue
        good.append((sc, p, st))
        if len(good) >= 6:
            break
    if not good:
        return random.choice(list(data)), 0.0
    wts = [max(0.05, g[0] - good[-1][0] + 0.1) for g in good]
    g = random.choices(good, weights=wts)[0]
    return g[1], g[2]

def mean_volume(path, start, dur):
    r = subprocess.run(["ffmpeg", "-ss", f"{start:.2f}", "-t", f"{dur}", "-i", path,
                        "-vn", "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"mean_volume: (-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None

ENC = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
log = []

# ---------- intro ----------
icands = candidates(INTRO, set())
chosen, seen = None, set()
for sc, p, st in icands:
    if p in seen:
        continue
    seen.add(p)
    vol = mean_volume(p, st, INTRO)
    if chosen is None:
        chosen = (p, st, vol)
    if vol is not None and vol > -35:
        chosen = (p, st, vol)
        break
    if len(seen) >= 8:
        break
if chosen is None:
    p0 = random.choice(list(data))
    chosen = (p0, 0.0, None)
ip, ist, ivol = chosen
log.append(f"INTRO: {os.path.basename(ip)} @ {ist:.1f}s")
base = "crop=ih*9/16:ih,scale=1080:1920,eq=saturation=0.75:brightness=-0.05"
zoom = ("scale=w='trunc(1080*(1+0.04*t)/2)*2':h='trunc(1920*(1+0.04*t)/2)*2':"
        "eval=frame,crop=1080:1920")
for vf in (f"{base},{zoom},setsar=1,fps=30", f"{base},setsar=1,fps=30"):
    try:
        ff(["-ss", f"{ist:.2f}", "-t", f"{INTRO}", "-i", ip, "-vf", vf] + ENC + ["tmp/seg_intro.mp4"])
        break
    except subprocess.CalledProcessError:
        print("intro zoom failed, trying plain")
if ivol is not None:
    ff(["-ss", f"{ist:.2f}", "-t", f"{INTRO}", "-i", ip, "-vn", "-ac", "2", "-ar", "44100",
        "-af", "loudnorm=I=-14:TP=-1.5", "tmp/intro_a.wav"])
else:
    ff(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{INTRO}", "tmp/intro_a.wav"])

# ---------- song-part segments ----------
def seg_filter(style, slow, flash):
    pre = "setpts=2.0*PTS," if slow else ""
    if style == "blur":
        return ("[0:v]" + pre + "split[a][b];"
                "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
                "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30[v]")
    z = "crop=iw/1.15:ih/1.15," if style == "punch" else ""
    f = ",fade=t=in:st=0:d=0.1:color=white" if flash else ""
    return f"[0:v]{pre}crop=ih*9/16:ih,{z}scale=1080:1920,setsar=1,fps=30{f}[v]"

parts, recent, used, hold_at = ["tmp/seg_intro.mp4"], [], [], None
for i, sg in enumerate(segs):
    L = sg["b"] - sg["a"]
    src = L / 2 if sg["slow"] else L
    p_, st = pick(src, set(recent[-3:]), used)
    p = sg["a"] / SONGLEN
    if sg["slow"]:
        style, hold_at = "blur", sg["a"]
    else:
        style = "punch" if i % 2 else "fill"
    flash = (0.25 <= p < 0.9) and i % 4 == 3 and not sg["slow"]
    out = f"tmp/seg{i}.mp4"
    ff(["-ss", f"{st:.2f}", "-t", f"{src:.3f}", "-i", p_, "-filter_complex",
        seg_filter(style, sg["slow"], flash), "-map", "[v]"] + ENC + [out])
    parts.append(out)
    recent.append(p_)
    used.append((p_, st))
    log.append(f"{i}: {os.path.basename(p_)} @ {st:.1f}s ({style}{', slow' if sg['slow'] else ''})")

with open("tmp/list.txt", "w") as f:
    for p in parts:
        f.write("file '" + os.path.basename(p) + "'\n")
ff(["-f", "concat", "-safe", "0", "-i", "tmp/list.txt", "-c", "copy", "tmp/video.mp4"])

# ---------- captions ----------
pool = []
if os.path.exists("captions.txt"):
    for line in open("captions.txt"):
        line = line.strip()
        if line and not line.startswith("#") and "|" in line:
            m, t = line.split("|", 1)
            if m.strip().lower() in (MOOD, "any"):
                pool.append(t.strip())
random.shuffle(pool)
caps_text = []
if OVERRIDE:
    caps_text.append(OVERRIDE)
elif pool:
    caps_text.append(pool.pop())
if pool and hold_at is not None:
    caps_text.append(pool.pop())

def write_cap(i, text):
    text = re.sub(r"[%\\]", "", text).upper()
    body = "\n".join(textwrap.fill(x.strip(), 16) for x in text.split("/"))
    path = f"tmp/cap{i}.txt"
    open(path, "w").write(body)
    return path

def dt(path, t0, t1, y, size):
    a = (f"if(lt(t,{t0:.2f}),0,if(lt(t,{t0 + 0.35:.2f}),(t-{t0:.2f})/0.35,"
         f"if(lt(t,{t1 - 0.35:.2f}),1,if(lt(t,{t1:.2f}),({t1:.2f}-t)/0.35,0))))")
    f = (f"drawtext=textfile={path}:expansion=none:fontcolor=white:fontsize={size}:"
         f"borderw=6:bordercolor=black:line_spacing=14:x=(w-text_w)/2:y={y}:alpha='{a}'")
    if FONT:
        f += f":fontfile={FONT}"
    return f

caps = []
if caps_text:
    caps.append(dt(write_cap(0, caps_text[0]), 0.5, INTRO - 0.2, "(h-text_h)/2-80", 84))
if len(caps_text) > 1:
    t0 = INTRO + hold_at
    caps.append(dt(write_cap(1,
