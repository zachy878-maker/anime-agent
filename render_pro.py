import glob, json, math, os, random, re, subprocess
import numpy as np
import librosa
from PIL import Image

E = os.environ.get
def flag(name, default="0"):
    return (E(name) or default) == "1"

STYLE = (E("STYLE") or "romance").strip().lower()
VERT = (E("FORMAT") or "landscape").strip().lower().startswith("v")
W, H = (1080, 1920) if VERT else (1920, 1080)
FPS = 30
DUR = float(E("DURATION") or 0)
START = float(E("START") or 0)
PARALLAX, VOICE, ISOLATE = flag("PARALLAX"), flag("VOICE", "1"), flag("ISOLATE")
LOOK_STRENGTH = float(E("LOOK_STRENGTH") or 0.7)
os.makedirs("out", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS)]

def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)

def ms(t):
    return int(round(max(0.0, t) * 1000))

def read_list(path):
    if not os.path.exists(path):
        return []
    return [l.strip().lower() for l in open(path) if l.strip() and not l.startswith("#")]

banned, favs = read_list("banned.txt"), read_list("favorites.txt")
CLIPS = [c for c in sorted(glob.glob("clips/*.mp4"))
         if not any(b in os.path.basename(c).lower() for b in banned)]
SONGS = [p for p in glob.glob("song.*") if p.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg"))]
if not CLIPS or not SONGS:
    raise SystemExit("need clips and a song.mp3 in the repo")
song = SONGS[0]

# ---------- style ----------
BASE = {"lo": 4.0, "mid": 2.2, "hi": 1.4, "xf": 12, "sat": 1.08, "warm": 0.04, "glow": 0.15,
        "zoom": 0.06, "trans": ["fade", "fade", "fade", "dissolve", "fadeblack"],
        "score": "romance", "particles": "petals"}
PRESETS = {
    "romance": {},
    "hype": {"lo": 1.6, "mid": 0.9, "hi": 0.6, "xf": 6, "sat": 1.15, "warm": 0.0, "glow": 0.08,
             "zoom": 0.1, "trans": ["fade", "slideleft", "zoomin", "wipeleft", "radial"],
             "score": "hype", "particles": "sparks"},
    "cinematic": {"lo": 5.0, "mid": 3.2, "hi": 2.0, "xf": 10, "sat": 0.95, "warm": -0.01,
                  "glow": 0.1, "zoom": 0.05, "trans": ["fade", "fadeblack", "fade"], "particles": "dust"},
}
ST = dict(BASE)
ST.update(PRESETS.get(STYLE, {}))
if os.path.exists(f"styles/{STYLE}.json"):
    ST.update(json.load(open(f"styles/{STYLE}.json")))
XF = int(ST["xf"])
MINF = XF * 2 + 6
print("style:", STYLE, "| format:", "vertical" if VERT else "landscape")

# ---------- colour look transfer ----------
M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
              [0.0193339, 0.1191920, 0.9503041]])
MINV = np.linalg.inv(M)
WHITE = np.array([0.95047, 1.0, 1.08883])

def rgb2lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ M.T) / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)

def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * WHITE
    lin = np.clip(xyz @ MINV.T, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)

def img_lab(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((320, 320))
    return rgb2lab(np.asarray(im, dtype=np.float32).reshape(-1, 3) / 255)

REF = None
for pth in glob.glob("look.*"):
    if pth.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        lab = img_lab(pth)
        REF = (lab.mean(0), lab.std(0) + 1e-3)
        print("look reference:", pth)
        break

def build_lut(src_png, path):
    if REF is None:
        return None
    lab = img_lab(src_png)
    mu_s, sd_s = lab.mean(0), lab.std(0) + 1e-3
    scale = np.clip(REF[1] / sd_s, 0.6, 1.6)
    n = 33
    g = np.linspace(0, 1, n, dtype=np.float32)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).transpose(2, 1, 0, 3).reshape(-1, 3)
    glab = rgb2lab(grid)
    tgt = (glab - mu_s) * scale + REF[0]
    out = lab2rgb((1 - LOOK_STRENGTH) * glab + LOOK_STRENGTH * tgt)
    with open(path, "w") as f:
        f.write(f"LUT_3D_SIZE {n}\n")
        f.write("\n".join(f"{a:.5f} {b:.5f} {c:.5f}" for a, b, c in out))
    return path

# ---------- song, beats, energy, drops ----------
y, sr = librosa.load(song, sr=22050, offset=START, duration=(DUR if DUR > 0 else None))
nframes = int(len(y) / sr * FPS)
dur = nframes / FPS
if dur < 30:
    raise SystemExit("song excerpt too short")
env = librosa.onset.onset_strength(y=y, sr=sr)
_, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
bt = [float(t) for t in librosa.frames_to_time(beats, sr=sr)]
if len(bt) < 8:
    raise SystemExit("could not find a beat in this song")
HOP = 512
rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

def smooth(sec):
    k = max(1, int(sr / HOP * sec))
    return np.convolve(rms, np.ones(k) / k, mode="same")

slow, fast = smooth(4), smooth(0.5)
lo_t, hi_t = np.percentile(slow, 35), np.percentile(slow, 72)

def ix(t):
    return min(len(rms) - 1, max(0, int(t * sr / HOP)))

def find_drops():
    sc, t = [], 6.0
    while t < dur - 6:
        a, b = ix(t - 2.5), ix(t - 0.2)
        before = fast[a:b].mean() if b > a else 0.0
        after = fast[ix(t):ix(t + 2.0)].mean()
        sc.append((after / (before + 1e-6), t))
        t += 0.25
    out = []
    for r, t in sorted(sc, reverse=True):
        if r < 1.5 or len(out) >= 3:
            break
        if all(abs(t - o) > 15 for o in out):
            out.append(min(bt, key=lambda b: abs(b - t)))
    return sorted(out)

def target_len(t):
    e = float(slow[ix(t)])
    base = ST["lo"] if e < lo_t else (ST["hi"] if e >= hi_t else ST["mid"])
    return base * random.uniform(0.85, 1.15)

cuts = [0.0]
while True:
    goal = cuts[-1] + max(MINF / FPS, target_len(cuts[-1]))
    nxt = [b for b in bt if b >= goal - 0.12]
    if not nxt or nxt[0] > dur - 1.5:
        break
    cuts.append(nxt[0])
drops = find_drops()
rad = MINF / FPS + 0.05
for d in drops:
    cuts = [c for c in cuts if abs(c - d) > rad] + [d]
cuts = sorted(cuts)
frames = [0]
for c in cuts[1:]:
    f = int(round(c * FPS))
    if f - frames[-1] >= MINF and nframes - f >= MINF:
        frames.append(f)
frames.append(nframes)
N = len(frames) - 1
dframes = [int(round(d * FPS)) for d in drops]
drop_idx = {i for i in range(1, N) if any(abs(frames[i] - df) <= 2 for df in dframes)}
print("shots:", N, "| drops at:", [round(d, 1) for d in drops])

# ---------- clip analysis ----------
def analyze(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf",
           "fps=4,scale=160:-2,signalstats,metadata=mode=print:file=-", "-an", "-f", "null", "-"]
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
    if len(rows) < 12:
        return None
    g = lambda k, d: np.array([r.get(k, d) for r in rows], dtype=float)
    arr = {"mot": np.minimum(g("YDIF", 0) / 25, 1),
           "con": (g("YHIGH", 200) - g("YLOW", 16)) / 255,
           "sat": np.minimum(g("SATAVG", 30) / 60, 1),
           "yv": g("YAVG", 100),
           "warm": g("VAVG", 128) - g("UAVG", 128),
           "cut": (g("YDIF", 0) > 45).astype(float)}
    d = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in arr.items()}
    d["n"] = len(rows)
    return d

data = {}
for c in random.sample(CLIPS, min(70, len(CLIPS))):
    d = analyze(c)
    if d:
        data[c] = d
if not data:
    raise SystemExit("no usable clips")
print("analyzed clips:", len(data))

def wmean(d, k, i, w):
    return (d[k][i + w] - d[k][i]) / w

def candidates(L):
    w = max(2, int(round(L * 4)))
    hype = ST["score"] == "hype"
    out = []
    for path, d in data.items():
        if d["n"] < w + 2:
            continue
        bonus = 0.3 if any(f in os.path.basename(path).lower() for f in favs) else 0.0
        for i in range(0, d["n"] - w, 2):
            mot, yv = wmean(d, "mot", i, w), wmean(d, "yv", i, w)
            sat = wmean(d, "sat", i, w)
            if hype:
                sc = mot + 0.5 * wmean(d, "con", i, w) + 0.3 * sat + bonus
                if yv < 45:
                    sc -= 0.6
            else:
                warm = wmean(d, "warm", i, w)
                sc = (0.9 * sat + 1.2 * float(np.clip(warm / 60, -0.5, 1))
                      + 0.5 * max(0.0, 1 - abs(mot - 0.2) * 2.5) + bonus)
                if mot > 0.6:
                    sc -= (mot - 0.6) * 2
                if yv < 60:
                    sc -= 0.8
            if yv > 215:
                sc -= 0.4
            if d["cut"][i + w] - d["cut"][i + 1] > 0:
                sc -= 1.0
            out.append((sc, path, i * 0.25))
    out.sort(reverse=True)
    return out

recent, used = [], []
def pick(L):
    good = []
    for sc, p, st in candidates(L):
        if p in recent[-4:]:
            continue
        if any(p == up and abs(st - ust) < L + 1 for up, ust in used):
            continue
        good.append((sc, p, st))
        if len(good) >= 8:
            break
    if not good:
        return random.choice(list(data)), 0.0
    wts = [max(0.05, g[0] - good[-1][0] + 0.1) for g in good]
    g = random.choices(good, weights=wts)[0]
    return g[1], g[2]

AUD = {}
def has_audio(path):
    if path not in AUD:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=index", "-of", "csv=p=0", path], capture_output=True, text=True)
        AUD[path] = bool(r.stdout.strip())
    return AUD[path]

def mean_volume(path, start, d):
    r = subprocess.run(["ffmpeg", "-ss", f"{start:.2f}", "-t", f"{d}", "-i", path, "-vn",
                        "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"mean_volume: (-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None

def pick_dialogue(L):
    w = max(2, int(round(L * 4)))
    c = []
    for path, d in data.items():
        if d["n"] < w + 2 or not has_audio(path):
            continue
        for i in range(0, d["n"] - w, 2):
            mot = wmean(d, "mot", i, w)
            if mot > 0.35 or wmean(d, "yv", i, w) < 50 or d["cut"][i + w] - d["cut"][i + 1] > 0:
                continue
            c.append((1 - mot + 0.5 * wmean(d, "con", i, w), path, i * 0.25))
    c.sort(reverse=True)
    seen = set()
    for sc, p, st in c:
        if p in seen:
            continue
        seen.add(p)
        v = mean_volume(p, st, L)
        if v is not None and v > -30:
            return p, st
        if len(seen) >= 12:
            break
    return None

# ---------- shot builders ----------
def saliency(png):
    g = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
    col = np.convolve(np.abs(np.diff(g, axis=1)).sum(0), np.ones(15) / 15, mode="same")
    idx = np.where(col >= np.percentile(col, 70))[0]
    return float(idx.mean() / len(col)) if len(idx) else 0.5

def pan_for(p, st, L):
    try:
        xs = []
        for k, t in enumerate((st, st + max(0.1, L - 0.2))):
            fp = f"tmp/pan{k}.png"
            ff(["-ss", f"{t:.2f}", "-i", p, "-frames:v", "1", "-vf", "scale=320:-2", fp])
            xs.append(float(np.clip(saliency(fp), 0.25, 0.75)))
        return xs[0], xs[0] + float(np.clip(xs[1] - xs[0], -0.15, 0.15))
    except Exception:
        return 0.5, 0.5

def seg_graphs(L, pan, lut, shake, i):
    z = ST["zoom"]
    zx = f"(1+{z}*t/{L:.3f})" if i % 2 == 0 else f"(1+{z}-{z}*t/{L:.3f})"
    if VERT:
        base = (f"crop=w=ih*9/16:h=ih:x='clip(iw*({pan[0]:.3f}+({pan[1] - pan[0]:.3f})*t/{L:.3f})-ow/2,0,iw-ow)':y=0,"
                f"scale={W}:{H}:flags=lanczos,setsar=1,fps={FPS}")
    else:
        base = (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={W}:{H},setsar=1,fps={FPS}")
    pre = base
    if lut:
        pre += f",lut3d=file={lut}:interp=tetrahedral"
    pre += ",hqdn3d=1.5:1.5:3:3,unsharp=5:5:0.5:5:5:0.0"
    if shake:
        pre += (",crop=w=iw*0.96:h=ih*0.96:x='iw*0.02+iw*0.015*sin(t*70)*max(0,1-t*2.5)':"
                f"y='ih*0.02+ih*0.015*cos(t*53)*max(0,1-t*2.5)',scale={W}:{H}")
    grade = f"eq=saturation={ST['sat']}:gamma=1.03,colorbalance=rm={ST['warm']}:bm={-ST['warm']}"
    full = (f"[0:v]{pre},scale=w='trunc({W}*{zx}/2)*2':h='trunc({H}*{zx}/2)*2':eval=frame,crop={W}:{H},"
            f"{grade},split[a][b];[b]scale=480:-2,gblur=sigma=6,scale={W}:{H}[g];"
            f"[a][g]blend=all_mode=screen:all_opacity={ST['glow']},format=yuv420p[v]")
    mid = f"[0:v]{pre},{grade},format=yuv420p[v]"
    simple = f"[0:v]{base},{grade},format=yuv420p[v]"
    return [full, mid, simple]

def make_seg(p, st, nfr, i, out, shake):
    L = nfr / FPS
    pan = pan_for(p, st, L) if VERT else None
    lut = None
    try:
        fp = f"tmp/f{i}.png"
        ff(["-ss", f"{st + L / 2:.2f}", "-i", p, "-frames:v", "1", "-vf", "scale=320:-2", fp])
        lut = build_lut(fp, f"tmp/lut{i}.cube")
    except Exception as ex:
        print("look transfer skipped:", str(ex)[:80])
    for fc in seg_graphs(L, pan, lut, shake, i):
        try:
            ff(["-ss", f"{st:.2f}", "-t", f"{L + 0.3:.3f}", "-i", p, "-filter_complex", fc,
                "-map", "[v]", "-frames:v", str(nfr), "-an"] + VENC + [out])
            return
        except subprocess.CalledProcessError:
            print("effects reduced on shot", i)
    raise SystemExit(f"shot {i} failed")

def hero_seg(p, st, nfr, out, i):
    import fx3d
    L = nfr / FPS
    png = f"tmp/hero{i}.png"
    vf = (f"crop=w=ih*9/16:h=ih,scale={W}:{H}:flags=lanczos" if VERT else
          f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}")
    ff(["-ss", f"{st + L / 2:.2f}", "-i", p, "-frames:v", "1", "-vf", vf, png])
    raw = f"tmp/hero_raw{i}.mp4"
    fx3d.hero(png, nfr, raw, W, H, FPS, ST["particles"])
    lut = None
    try:
        lut = build_lut(png, f"tmp/lut{i}.cube")
    except Exception:
        pass
    vf2 = (f"lut3d=file={lut}:interp=tetrahedral," if lut else "") + f"eq=saturation={ST['sat']},format=yuv420p"
    ff(["-i", raw, "-vf", vf2, "-frames:v", str(nfr), "-an"] + VENC + [out])

def extract_voice(p, st, L):
    vlen = min(2.6, L - 0.3)
    ff(["-ss", f"{st:.2f}", "-t", f"{vlen:.2f}", "-i", p, "-vn", "-ac", "2", "-ar", "44100", "tmp/voice_raw.wav"])
    src = "tmp/voice_raw.wav"
    if ISOLATE:
        try:
            subprocess.run(["python", "-m", "demucs", "--two-stems=vocals", "-o", "tmp/sep", src],
                           check=True, capture_output=True)
            found = glob.glob("tmp/sep/*/voice_raw/vocals.wav")
            if found:
                src = found[0]
                print("voice isolated")
        except Exception:
            print("voice isolation failed, using the raw audio")
    ff(["-i", src, "-af", f"loudnorm=I=-16:TP=-1.5,afade=t=in:d=0.05,afade=t=out:st={vlen - 0.2:.2f}:d=0.2",
        "-ar", "44100", "-ac", "2", "tmp/voice.wav"])
    return vlen

# ---------- plan and render shots ----------
vj = -1
if VOICE and drop_idx:
    cand = min(drop_idx) - 1
    if cand >= 0 and (frames[cand + 1] - frames[cand]) / FPS >= 1.6:
        vj = cand
heroes = set()
if PARALLAX:
    last = -99
    for i in range(1, N - 1):
        if i in drop_idx or i == vj:
            continue
        if frames[i + 1] - frames[i] >= 72 and i - last >= max(5, N // 5) and len(heroes) < 4:
            heroes.add(i)
            last = i

segfiles, log, voice_info = [], [], None
for i in range(N):
    D = frames[i + 1] - frames[i]
    nfr = D + XF if i < N - 1 else D
    L = nfr / FPS
    dlg = pick_dialogue(L) if i == vj else None
    p, st = dlg or pick(L + 0.2)
    out = f"tmp/seg{i}.mp4"
    done = False
    if i in heroes:
        try:
            hero_seg(p, st, nfr, out, i)
            done = True
            log.append(f"{i}: {os.path.basename(p)} @ {st:.1f}s (2.5D parallax)")
        except Exception as ex:
            print("parallax failed, using a normal shot:", str(ex)[:150])
    if not done:
        make_seg(p, st, nfr, i, out, i in drop_idx)
        tag = " (DROP)" if i in drop_idx else ""
        log.append(f"{i}: {os.path.basename(p)} @ {st:.1f}s{tag}")
    if dlg:
        try:
            vl = extract_voice(p, st, L)
            voice_info = (frames[i] / FPS + 0.2, vl)
            log[-1] += " (voice line)"
        except Exception as ex:
            print("voice tag skipped:", str(ex)[:100])
    segfiles.append(out)
    recent.append(p)
    used.append((p, st))
    print("shot", i + 1, "of", N)

# ---------- join with transitions ----------
trans = [None] + ["fadewhite" if i in drop_idx else random.choice(ST["trans"]) for i in range(1, N)]

def join(tr):
    inputs, chain, prev = [], [], "0:v"
    for s in segfiles:
        inputs += ["-i", s]
    for i in range(1, N):
        chain.append(f"[{prev}][{i}:v]xfade=transition={tr[i]}:duration={XF / FPS:.3f}:"
                     f"offset={frames[i] / FPS:.3f}[x{i}]")
        prev = f"x{i}"
    ff(inputs + ["-filter_complex", ";".join(chain), "-map", f"[{prev}]", "-t", f"{dur:.3f}"]
       + VENC + ["tmp/video.mp4"])

try:
    join(trans)
except subprocess.CalledProcessError:
    print("transitions failed, using plain crossfades")
    join([None] + ["fade"] * (N - 1))

# ---------- lyrics (your file, or auto) as animated captions ----------
def parse_srt(txt):
    out = []
    for block in re.split(r"\n\s*\n", txt.replace("\r", "").strip()):
        lines = [l for l in block.split("\n") if l.strip()]
        for j, l in enumerate(lines):
            m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", l.strip())
            if m:
                g = [int(x) for x in m.groups()]
                out.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                            g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                            " ".join(x.strip() for x in lines[j + 1:])))
                break
    return out

def parse_lrc(txt):
    items = []
    for line in txt.splitlines():
        m = re.match(r"\s*\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if m and m.group(3).strip():
            items.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    return [(s, min(items[i + 1][0] - 0.05 if i + 1 < len(items) else s + 5, s + 8), t)
            for i, (s, t) in enumerate(items)]

def tc(t):
    v = ms(t)
    return f"{v // 3600000:02d}:{v // 60000 % 60:02d}:{v // 1000 % 60:02d},{v % 1000:03d}"

def transcribe():
    from faster_whisper import WhisperModel
    ff(["-ss", f"{START:.2f}", "-t", f"{dur:.3f}", "-i", song, "-vn", "-ac", "1", "-ar", "16000", "tmp/excerpt.wav"])
    model = WhisperModel(E("WHISPER_MODEL") or "small", device="cpu", compute_type="int8")
    segs, info = model.transcribe("tmp/excerpt.wav", language=(E("LYRICS_LANG") or None),
                                  vad_filter=True, condition_on_previous_text=False,
                                  beam_size=5, word_timestamps=True)
    rows, last = [], ""
    for s in segs:
        t = s.text.strip()
        if not t or s.no_speech_prob > 0.6 or s.avg_logprob < -1.3 or t.lower() == last.lower():
            continue
        last = t
        words = [(w.start, w.end, w.word.strip()) for w in (s.words or []) if w.word.strip()]
        rows.append((max(0.0, s.start), min(dur, s.end), t, words or None))
    print("language:", info.language, "| lines found:", len(rows))
    return rows

def ass_time(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"

def esc(t):
    return re.sub(r"[{}\\]", "", t).replace("\n", " ").strip()

def write_ass(rows, path):
    fs, mv = (66, 260) if VERT else (58, 70)
    head = ("[Script Info]\nScriptType: v4.00+\n"
            f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
            "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Lyric,DejaVu Sans,{fs},&H00FFFFFF,&H00D0B0FF,&H00101010,&H96000000,"
            f"1,0,0,0,100,100,1,0,1,3,2,2,60,60,{mv},1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    lines = []
    for s, e, text, words in rows:
        anim = "{\\fad(220,260)\\blur6\\fscx118\\fscy118\\t(0,260,\\blur0\\fscx100\\fscy100)"
        for b in [b for b in bt if s + 0.35 < b < e - 0.25][::2][:14]:
            r = int((b - s) * 1000)
            anim += f"\\t({r},{r + 90},\\fscx106\\fscy106)\\t({r + 90},{r + 260},\\fscx100\\fscy100)"
        anim += "}"
        if words:
            body = ""
            for j, (ws, we, w) in enumerate(words):
                nx = words[j + 1][0] if j + 1 < len(words) else we
                body += "{\\kf" + str(max(1, int((nx - ws) * 100))) + "}" + esc(w) + " "
        else:
            body = esc(text)
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Lyric,,0,0,0,,{anim}{body.strip()}")
    open(path, "w", encoding="utf-8").write(head + "\n".join(lines) + "\n")

def build_lyrics():
    rows, entries = [], None
    for name, parser in (("lyrics.srt", parse_srt), ("lyrics.lrc", parse_lrc)):
        if os.path.exists(name):
            entries = parser(open(name, encoding="utf-8", errors="ignore").read())
            break
    if entries:
        for s, e, t in entries:
            s2, e2 = max(0.0, s - START), min(dur, e - START)
            if e2 - s2 > 0.2 and t:
                rows.append((s2, e2, t, None))
    elif (E("AUTO_LYRICS") or "1") != "0":
        print("no lyrics file found, transcribing the song...")
        try:
            raw = transcribe()
        except Exception as ex:
            print("transcription failed:", str(ex)[:200])
            raw = []
        for i, r in enumerate(raw):
            e = min(r[1], raw[i + 1][0] - 0.05) if i + 1 < len(raw) else r[1]
            if e - r[0] > 0.2:
                rows.append((r[0], e, r[2], r[3]))
        if rows:
            with open("out/lyrics_auto.srt", "w", encoding="utf-8") as f:
                for i, (s, e, t, _) in enumerate(rows, 1):
                    f.write(f"{i}\n{tc(s + START)} --> {tc(e + START)}\n{t}\n\n")
    if not rows:
        return False
    write_ass(rows, "tmp/lyrics.ass")
    print("lyric lines:", len(rows))
    return True

has_ass = build_lyrics()

# ---------- audio design + final mix ----------
def audio_graph():
    fmt = "aformat=sample_rates=44100:channel_layouts=stereo"
    inputs = ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    parts, labels, dips, k = [], [], [], 3
    for d in drops:
        for src, length, chain in (
                ("anoisesrc=c=white:a=0.5:d=2.0", 2.0,
                 f"highpass=f=900,afade=t=in:st=0:d=2.0,volume=0.6,adelay={ms(d - 2.0)}|{ms(d - 2.0)}"),
                ("sine=frequency=46:sample_rate=44100:duration=1.6", 1.6,
                 f"afade=t=out:st=0.05:d=1.5,volume=5,adelay={ms(d)}|{ms(d)}"),
                ("anoisesrc=c=pink:a=0.6:d=1.2", 1.2,
                 f"lowpass=f=4500,afade=t=out:st=0:d=1.2,volume=0.8,adelay={ms(d)}|{ms(d)}")):
            inputs += ["-f", "lavfi", "-t", f"{length}", "-i", src]
            parts.append(f"[{k}:a]{fmt},{chain}[e{k}]")
            labels.append(f"[e{k}]")
            k += 1
        dips.append((d - 0.3, d, 0.08))
    if voice_info:
        vs, vl = voice_info
        inputs += ["-i", "tmp/voice.wav"]
        parts.append(f"[{k}:a]{fmt},volume=1.5,adelay={ms(vs)}|{ms(vs)}[e{k}]")
        labels.append(f"[e{k}]")
        dips.append((vs, vs + vl, 0.3))
    expr = "*".join(f"if(between(t,{a:.2f},{b:.2f}),{g},1)" for a, b, g in dips) or "1"
    parts.append(f"[1:a]{fmt},volume='{expr}':eval=frame,afade=t=in:d=1,afade=t=out:st={dur - 4:.2f}:d=4[sg]")
    if labels:
        parts.append(f"[2:a]{''.join(labels)}amix=inputs={len(labels) + 1}:duration=longest:normalize=0,"
                     f"atrim=0:{dur:.3f}[bus]")
        parts.append("[bus]asplit=2[sc][bm]")
        parts.append("[sg][sc]sidechaincompress=threshold=0.06:ratio=5:attack=8:release=350[duck]")
        parts.append("[duck][bm]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5[a]")
    else:
        parts.append("[sg]loudnorm=I=-14:TP=-1.5[a]")
    return inputs, parts

def run_final(use_ass, use_fx):
    inputs = ["-i", "tmp/video.mp4", "-ss", f"{START:.2f}", "-t", f"{dur:.3f}", "-i", song]
    vf = ["ass=tmp/lyrics.ass:fontsdir=/usr/share/fonts/truetype/dejavu"] if use_ass else []
    vf.append(f"fade=t=in:st=0:d=1.5,fade=t=out:st={dur - 3:.2f}:d=3")
    parts = [f"[0:v]{','.join(vf)}[v]"]
    if use_fx:
        extra, ap = audio_graph()
        inputs += extra
        parts += ap
    else:
        parts.append(f"[1:a]afade=t=in:d=1,afade=t=out:st={dur - 4:.2f}:d=4,loudnorm=I=-14:TP=-1.5[a]")
    ff(inputs + ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", "-movflags", "+faststart",
                 "out/edit.mp4"])

tries = []
for combo in ((has_ass, True), (False, True), (has_ass, False), (False, False)):
    if combo not in tries:
        tries.append(combo)
for use_ass, use_fx in tries:
    try:
        run_final(use_ass, use_fx)
        print("final render ok | lyrics:", use_ass, "| sound design:", use_fx)
        break
    except subprocess.CalledProcessError:
        print("final render failed with lyrics =", use_ass, "sound design =", use_fx, "- trying simpler")
else:
    raise SystemExit("final render failed")

with open("credit.txt", "w") as f:
    f.write(f"STYLE: {STYLE} | drops at: {[round(d, 1) for d in drops]}\n")
    f.write("CLIPS USED (add bad ones to banned.txt):\n" + "\n".join(log) + "\n")
print("done: out/edit.mp4,", round(dur, 1), "seconds")
