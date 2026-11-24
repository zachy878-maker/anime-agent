import glob, os, random, re, subprocess, textwrap
import numpy as np

W, H = 1920, 1080
ANIME = (os.environ.get("ANIME") or "Attack_on_Titan").strip()
TITLE = (os.environ.get("TITLE") or ANIME.replace("_", " ")).strip().upper()
DIALOGUES = max(1, min(3, int(os.environ.get("DIALOGUES") or 2)))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(FONT):
    FONT = None
os.makedirs("out", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-r", "30"]
AENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]

def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)

def read_list(path):
    if not os.path.exists(path):
        return []
    return [l.strip().lower() for l in open(path) if l.strip() and not l.startswith("#")]

banned, favs = read_list("banned.txt"), read_list("favorites.txt")
CLIPS = [c for c in sorted(glob.glob("clips/*.mp4"))
         if not any(b in os.path.basename(c).lower() for b in banned)]
if not CLIPS:
    raise SystemExit("no clips")

DEFAULT_CARDS = ["THE WORLD IS CRUEL", "BUT SOME REFUSE / TO KNEEL",
                 "THIS IS / THEIR WAR", "NO TURNING BACK"]
cards = []
if os.path.exists("cards.txt"):
    cards = [l.strip() for l in open("cards.txt") if l.strip() and not l.startswith("#")]
cards = cards or DEFAULT_CARDS

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
        "con": (g("YHIGH", 200) - g("YLOW", 16)) / 255,
        "sat": np.minimum(g("SATAVG", 30) / 60, 1),
        "yv": g("YAVG", 100),
        "cut": (g("YDIF", 0) > 45).astype(float),
    }
    d = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in arr.items()}
    d["n"] = len(rows)
    return d

data = {}
for c in random.sample(CLIPS, min(45, len(CLIPS))):
    d = analyze(c)
    if d:
        data[c] = d
if not data:
    raise SystemExit("no usable clips")
print("analyzed clips:", len(data))

def wmean(d, k, i, w):
    return (d[k][i + w] - d[k][i]) / w

def is_fav(path):
    return any(f in os.path.basename(path).lower() for f in favs)

def hype_candidates(L):
    w = max(2, int(round(L * 4)))
    out = []
    for path, d in data.items():
        if d["n"] < w + 2:
            continue
        bonus = 0.3 if is_fav(path) else 0.0
        for i in range(0, d["n"] - w, 2):
            yv = wmean(d, "yv", i, w)
            sc = (wmean(d, "mot", i, w) + 0.5 * wmean(d, "con", i, w)
                  + 0.3 * wmean(d, "sat", i, w) + bonus)
            if yv < 45:
                sc -= 0.6
            if yv > 200:
                sc -= 0.3
            if d["cut"][i + w] - d["cut"][i + 1] > 0:
                sc -= 1.0
            out.append((sc, path, i * 0.25))
    out.sort(reverse=True)
    return out

def dialogue_candidates(L):
    w = max(2, int(round(L * 4)))
    out = []
    for path, d in data.items():
        if d["n"] < w + 2:
            continue
        bonus = 0.3 if is_fav(path) else 0.0
        for i in range(0, d["n"] - w, 2):
            mot = wmean(d, "mot", i, w)
            yv = wmean(d, "yv", i, w)
            if d["cut"][i + w] - d["cut"][i + 1] > 0 or yv < 50 or mot > 0.4:
                continue
            out.append((1 - mot + 0.5 * wmean(d, "con", i, w) + bonus, path, i * 0.25))
    out.sort(reverse=True)
    return out

AUD = {}
def has_audio(path):
    if path not in AUD:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                            "-show_entries", "stream=index", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
        AUD[path] = bool(r.stdout.strip())
    return AUD[path]

def mean_volume(path, start, dur):
    r = subprocess.run(["ffmpeg", "-ss", f"{start:.2f}", "-t", f"{dur}", "-i", path,
                        "-vn", "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"mean_volume: (-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None

used_win, recent = [], []

def pick_hype(L):
    allc = hype_candidates(L)
    good = []
    for sc, p, st in allc:
        if p in recent[-2:]:
            continue
        if any(p == up and abs(st - ust) < L + 0.5 for up, ust in used_win):
            continue
        good.append((sc, p, st))
        if len(good) >= 6:
            break
    if not good:
        p = random.choice(list(data))
        return p, 0.0
    wts = [max(0.05, g[0] - good[-1][0] + 0.1) for g in good]
    g = random.choices(good, weights=wts)[0]
    return g[1], g[2]

dialogue_clips = set()

def pick_dialogue(L):
    cands = dialogue_candidates(L)
    checked, seen = [], set()
    for sc, p, st in cands:
        if p in dialogue_clips or p in seen or not has_audio(p):
            continue
        seen.add(p)
        vol = mean_volume(p, st, L)
        if vol is None:
            continue
        checked.append((vol, p, st))
        if vol > -30:
            return p, st
        if len(checked) >= 10:
            break
    if checked:
        best = max(checked)
        return best[1], best[2]
    return pick_hype(L)

# ---------- building blocks ----------
def clip_seg(path, start, outdur, out, vol=1.0, slow=False, grade="",
             zoom=False, flash=False, fin=0.0, fout=0.0):
    src = outdur / 2 if slow else outdur
    vf = ("setpts=2.0*PTS," if slow else "")
    vf += f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    if zoom:
        vf += f",crop=iw/1.12:ih/1.12,scale={W}:{H}"
    vf += ",setsar=1,fps=30" + grade
    if fin:
        vf += f",fade=t=in:st=0:d={fin}"
    if fout:
        vf += f",fade=t=out:st={outdur - fout:.2f}:d={fout}"
    if flash:
        vf += ",fade=t=in:st=0:d=0.1:color=white"
    native = (not slow) and vol > 0 and has_audio(path)
    args = ["-ss", f"{start:.2f}", "-t", f"{src:.3f}", "-i", path]
    if native:
        af = (f"aresample=44100,aformat=channel_layouts=stereo,volume={vol},"
              f"afade=t=in:d=0.03,afade=t=out:st={max(0, outdur - 0.06):.2f}:d=0.06")
        args += ["-vf", vf, "-af", af]
    else:
        args += ["-f", "lavfi", "-t", f"{outdur:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
                 "-vf", vf, "-map", "0:v", "-map", "1:a"]
    ff(args + ["-t", f"{outdur:.3f}"] + VENC + AENC + [out])

def cap_file(i, text):
    text = re.sub(r"[%\\]", "", text).upper()
    body = "\n".join(textwrap.fill(x.strip(), 22) for x in text.split("/"))
    path = f"tmp/cap{i}.txt"
    open(path, "w").write(body)
    return path

def drawtext(path, D, size):
    t0, t1 = 0.3, D - 0.25
    a = (f"if(lt(t,{t0}),0,if(lt(t,{t0 + 0.4:.2f}),(t-{t0})/0.4,"
         f"if(lt(t,{t1 - 0.4:.2f}),1,if(lt(t,{t1:.2f}),({t1:.2f}-t)/0.4,0))))")
    f = (f"drawtext=textfile={path}:expansion=none:fontcolor=white:fontsize={size}:"
         f"line_spacing=18:x=(w-text_w)/2:y=(h-text_h)/2:alpha='{a}'")
    if FONT:
        f += f":fontfile={FONT}"
    return f

def card_seg(i, text, D, out, size=110, freq=50, gain=3.0):
    base = ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=30:d={D}",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=44100:duration={D}"]
    af = (f"aformat=channel_layouts=stereo,volume={gain},"
          f"afade=t=in:d=0.01,afade=t=out:st=0.15:d={D - 0.15:.2f}")
    try:
        vf = drawtext(cap_file(i, text), D, size) + ",setsar=1"
        ff(base + ["-vf", vf, "-af", af, "-t", f"{D}"] + VENC + AENC + [out])
    except subprocess.CalledProcessError:
        print("text failed on card", i, "- using plain black")
        ff(base + ["-vf", "setsar=1", "-af", af, "-t", f"{D}"] + VENC + AENC + [out])

def black_seg(D, out):
    ff(["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=30:d={D}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{D}"]
       + VENC + AENC + [out])

# ---------- timeline ----------
parts, log, now = [], [], 0.0
def add(path, dur):
    global now
    parts.append(path)
    now += dur

GRADE = ",eq=saturation=0.8:brightness=-0.05,vignette=PI/5"
DLG = [7.0, 5.0, 4.5]
n = 0
for k in range(DIALOGUES):
    L = DLG[k]
    p, st = pick_dialogue(L)
    dialogue_clips.add(p)
    out = f"tmp/s{n}.mp4"
    clip_seg(p, st, L, out, vol=1.0, grade=GRADE,
             fin=1.0 if k == 0 else 0.3, fout=0.3)
    add(out, L)
    log.append(f"DIALOGUE {k + 1}: {os.path.basename(p)} @ {st:.1f}s")
    n += 1
    out = f"tmp/s{n}.mp4"
    card_seg(k, cards[k % len(cards)], 2.4, out)
    add(out, 2.4)
    n += 1

MONTAGE = [(2.4, "n"), (2.0, "n"), (1.6, "n"), (1.3, "n"), (1.0, "n"), (0.8, "n"),
           (0.7, "n"), (0.6, "n"), (0.5, "n"), (2.0, "slow"), (0.4, "n"), (0.4, "n"),
           (0.35, "n"), (0.35, "n"), (0.3, "n"), (0.3, "n"), (0.3, "n"), (0.3, "n")]
T0 = now
for j, (L, kind) in enumerate(MONTAGE):
    slow = kind == "slow"
    p, st = pick_hype(L / 2 if slow else L)
    out = f"tmp/s{n}.mp4"
    clip_seg(p, st, L, out, vol=0.45, slow=slow,
             zoom=(j % 2 == 1 and not slow), flash=(j >= 9 and j % 4 == 1 and not slow))
    add(out, L)
    recent.append(p)
    used_win.append((p, st))
    log.append(f"{j}: {os.path.basename(p)} @ {st:.1f}s{' (slow-mo)' if slow else ''}")
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
