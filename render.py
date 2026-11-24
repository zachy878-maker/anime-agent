import glob, os, random, re, subprocess
import numpy as np
import librosa

W, H, FPS, XF = 1920, 1080, 30, 12
DUR = float(os.environ.get("DURATION") or 150)
START = float(os.environ.get("START") or 0)
os.makedirs("out", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS)]

def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)

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

# ---------- song, beats, energy ----------
y, sr = librosa.load(song, sr=22050, offset=START, duration=DUR)
nframes = int(len(y) / sr * FPS)
dur = nframes / FPS
if dur < 30:
    raise SystemExit("song excerpt too short")
env = librosa.onset.onset_strength(y=y, sr=sr)
_, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
bt = [float(t) for t in librosa.frames_to_time(beats, sr=sr)]
rms = librosa.feature.rms(y=y, hop_length=512)[0]
k = max(1, int(sr / 512 * 4))
sm = np.convolve(rms, np.ones(k) / k, mode="same")
lo, hi = np.percentile(sm, 35), np.percentile(sm, 72)

def energy(t):
    return float(sm[min(len(sm) - 1, int(t * sr / 512))])

cuts = [0.0]
while True:
    e = energy(cuts[-1])
    n = 8 if e < lo else (2 if e >= hi else 4)
    nxt = [b for b in bt if b > cuts[-1] + 0.3]
    if len(nxt) < n:
        break
    t = nxt[n - 1]
    while t - cuts[-1] < 1.3 and n < len(nxt):
        n += 1
        t = nxt[n - 1]
    if t > dur - 1.5:
        break
    cuts.append(t)

frames = [0]
for c in cuts[1:]:
    f = int(round(c * FPS))
    if f - frames[-1] >= 36:
        frames.append(f)
frames.append(nframes)
N = len(frames) - 1
print("song:", song, "| seconds:", round(dur, 1), "| shots:", N)

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
    if len(rows) < 12:
        return None
    g = lambda k, d: np.array([r.get(k, d) for r in rows], dtype=float)
    arr = {
        "mot": np.minimum(g("YDIF", 0) / 25, 1),
        "sat": np.minimum(g("SATAVG", 30) / 60, 1),
        "yv": g("YAVG", 100),
        "warm": g("VAVG", 128) - g("UAVG", 128),
        "cut": (g("YDIF", 0) > 45).astype(float),
    }
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
    out = []
    for path, d in data.items():
        if d["n"] < w + 2:
            continue
        bonus = 0.3 if any(f in os.path.basename(path).lower() for f in favs) else 0.0
        for i in range(0, d["n"] - w, 2):
            mot = wmean(d, "mot", i, w)
            yv = wmean(d, "yv", i, w)
            warm = wmean(d, "warm", i, w)
            sc = (0.9 * wmean(d, "sat", i, w) + 1.2 * float(np.clip(warm / 60, -0.5, 1))
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

# ---------- shots ----------
def make_seg(path, start, nfr, idx, out):
    L = nfr / FPS
    z = 0.06
    zx = f"(1+{z}*t/{L:.3f})" if idx % 2 == 0 else f"(1+{z}-{z}*t/{L:.3f})"
    base = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},setsar=1,fps={FPS}")
    grade = "eq=saturation=1.08:gamma=1.03,colorbalance=rm=0.04:bm=-0.04"
    full = (f"[0:v]{base},"
            f"scale=w='trunc({W}*{zx}/2)*2':h='trunc({H}*{zx}/2)*2':eval=frame,crop={W}:{H},"
            f"{grade},split[a][b];"
            f"[b]scale=480:-2,gblur=sigma=6,scale={W}:{H}[g];"
            f"[a][g]blend=all_mode=screen:all_opacity=0.15,format=yuv420p[v]")
    simple = f"[0:v]{base},{grade},format=yuv420p[v]"
    for fc in (full, simple):
        try:
            ff(["-ss", f"{start:.2f}", "-t", f"{L + 0.3:.3f}", "-i", path,
                "-filter_complex", fc, "-map", "[v]", "-frames:v", str(nfr), "-an"]
               + VENC + [out])
            return
        except subprocess.CalledProcessError:
            print("effects failed on shot", idx, "- trying plain")
    raise SystemExit("shot failed")

segfiles, log = [], []
for i in range(N):
    D = frames[i + 1] - frames[i]
    nfr = D + XF if i < N - 1 else D
    p, st = pick(nfr / FPS + 0.2)
    out = f"tmp/seg{i}.mp4"
    make_seg(p, st, nfr, i, out)
    segfiles.append(out)
    recent.append(p)
    used.append((p, st))
    log.append(f"{i}: {os.path.basename(p)} @ {st:.1f}s")
    print("shot", i + 1, "of", N)

# ---------- crossfade everything together ----------
if N == 1:
    ff(["-i", segfiles[0]] + VENC + ["tmp/video.mp4"])
else:
    inputs, chain, prev = [], [], "0:v"
    for i, s in enumerate(segfiles):
        inputs += ["-i", s]
    for i in range(1, N):
        lab = f"x{i}"
        chain.append(f"[{prev}][{i}:v]xfade=transition=fade:"
                     f"duration={XF / FPS:.3f}:offset={frames[i] / FPS:.3f}[{lab}]")
        prev = lab
    ff(inputs + ["-filter_complex", ";".join(chain), "-map", f"[{prev}]",
                 "-t", f"{dur:.3f}"] + VENC + ["tmp/video.mp4"])

# ---------- lyrics (from your own lyrics.srt or lyrics.lrc) ----------
def parse_srt(txt):
    out = []
    for block in re.split(r"\n\s*\n", txt.replace("\r", "").strip()):
        lines = [l for l in block.split("\n") if l.strip()]
        for j, l in enumerate(lines):
            m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", l.strip())
            if m:
                g = [int(x) for x in m.groups()]
                s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                out.append((s, e, " ".join(x.strip() for x in lines[j + 1:])))
                break
    return out

def parse_lrc(txt):
    items = []
    for line in txt.splitlines():
        m = re.match(r"\s*\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if m and m.group(3).strip():
            items.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    out = []
    for i, (s, t) in enumerate(items):
        e = items[i + 1][0] - 0.05 if i + 1 < len(items) else s + 5
        out.append((s, min(e, s + 8), t))
    return out

def tc(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

def build_lyrics():
    entries = None
    for name, parser in (("lyrics.srt", parse_srt), ("lyrics.lrc", parse_lrc)):
        if os.path.exists(name):
            entries = parser(open(name, encoding="utf-8", errors="ignore").read())
            break
    if not entries:
        return False
    rows = []
    for s, e, t in entries:
        s2, e2 = max(0.0, s - START), min(dur, e - START)
        if e2 - s2 > 0.2 and t:
            rows.append((s2, e2, t))
    if not rows:
        return False
    with open("tmp/lyrics.srt", "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(rows, 1):
            f.write(f"{i}\n{tc(s)} --> {tc(e)}\n{t}\n\n")
    print("lyric lines:", len(rows))
    return True

has_lyrics = build_lyrics()

# ---------- final: lyrics + fades + song ----------
fade = f"fade=t=in:st=0:d=1.5,fade=t=out:st={dur - 3:.2f}:d=3"
style = ("FontName=DejaVu Sans,FontSize=17,Bold=1,Alignment=2,MarginV=40,"
         "Outline=2,Shadow=1,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&")

def final(vf):
    ff(["-i", "tmp/video.mp4", "-ss", f"{START:.2f}", "-t", f"{dur:.3f}", "-i", song,
        "-vf", vf,
        "-af", f"afade=t=in:d=1,afade=t=out:st={dur - 4:.2f}:d=4,loudnorm=I=-14:TP=-1.5",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}",
        "-movflags", "+faststart", "out/edit.mp4"])

if has_lyrics:
    try:
        final(f"subtitles=tmp/lyrics.srt:force_style='{style}',{fade}")
    except subprocess.CalledProcessError:
        print("lyrics failed, rendering without them")
        final(fade)
else:
    print("no lyrics.srt or lyrics.lrc found, rendering without lyrics")
    final(fade)

with open("credit.txt", "w") as f:
    f.write("CLIPS USED (add bad ones to banned.txt):\n" + "\n".join(log) + "\n")
print("done: out/edit.mp4,", round(dur, 1), "seconds")p)} @ {st:.1f}s{' (slow-mo)' if slow else ''}")
    n += 1
TM = now - T0

out = f"tmp/s{n}.mp4"
black_seg(0.6, out)
add(out, 0.6)
n += 1
out = f"tmp/s{n}.mp4"
card_seg(9, TITLE, 4.2, out, size=130, freq=45, gain=4.0)
add(out, 4.2)
TT = now

# ---------- join ----------
with open("tmp/list.txt", "w") as f:
    for p in parts:
        f.write("file '" + os.path.basename(p) + "'\n")
ff(["-f", "concat", "-safe", "0", "-i", "tmp/list.txt"] + VENC + AENC + ["tmp/joined.mp4"])

# ---------- sound design: riser under the montage + low drone ----------
d0 = int(T0 * 1000)
fc = (f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,"
      f"afade=t=in:st=0:d={TM:.2f},highpass=f=300,volume=0.55,adelay={d0}|{d0}[r];"
      f"[2:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=1.2,"
      f"afade=t=in:d=3,afade=t=out:st={TT - 2.5:.2f}:d=2.5[d];"
      f"[0:a][r][d]amix=inputs=3:duration=first:normalize=0,"
      f"loudnorm=I=-14:TP=-1.5[a]")
try:
    ff(["-i", "tmp/joined.mp4",
        "-f", "lavfi", "-t", f"{TM:.2f}", "-i", f"anoisesrc=c=pink:a=0.5:d={TM:.2f}",
        "-f", "lavfi", "-t", f"{TT:.2f}", "-i", f"sine=frequency=55:sample_rate=44100:duration={TT:.2f}",
        "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy"]
       + AENC + ["-t", f"{TT:.2f}", "out/edit.mp4"])
except subprocess.CalledProcessError:
    print("sound design failed, using plain audio")
    ff(["-i", "tmp/joined.mp4", "-c", "copy", "out/edit.mp4"])

with open("credit.txt", "w") as f:
    f.write("TRAILER CLIPS (add bad ones to banned.txt):\n" + "\n".join(log) + "\n")
print("done: out/edit.mp4,", round(TT, 1), "seconds")\n".join(textwrap.fill(x.strip(), 16) for x in text.split("/"))
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
    caps.append(dt(write_cap(1, caps_text[1]), t0, t0 + 2.6, "h*0.68", 72))

# ---------- final mix ----------
D = int((INTRO - PRE) * 1000)
def final(with_caps):
    v = (caps if with_caps else []) + [f"fade=t=out:st={TOTALV - 1:.2f}:d=1"]
    fc = (f"[0:v]{','.join(v)}[v];"
          f"[1:a]afade=t=out:st={INTRO - 0.9:.2f}:d=0.9[ia];"
          f"[2:a]afade=t=in:st=0:d={PRE},adelay={D}|{D}[sa];"
          f"[ia][sa]amix=inputs=2:duration=longest:normalize=0,"
          f"loudnorm=I=-14:TP=-1.5,afade=t=out:st={TOTALV - 1:.2f}:d=1[a]")
    ff(["-i", "tmp/video.mp4", "-i", "tmp/intro_a.wav",
        "-ss", f"{offset - PRE:.2f}", "-t", f"{SONGLEN + PRE:.2f}", "-i", song,
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{TOTALV:.2f}", "out/edit.mp4"])

try:
    final(True)
except subprocess.CalledProcessError:
    print("captions failed, rendering without them")
    final(False)

# ---------- credit file ----------
name = os.path.basename(song)
with open("credit.txt", "w") as f:
    if os.path.exists("credits.csv"):
        for line in open("credits.csv"):
            if line.startswith(name + "|"):
                f.write("MUSIC: " + line)
    f.write("\nCLIPS USED (add bad ones to banned.txt):\n" + "\n".join(log) + "\n")
print("done: out/edit.mp4")
