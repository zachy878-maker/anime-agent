import glob, os, random, re, subprocess
import numpy as np
import librosa

DURATION = float(os.environ.get("DURATION") or 25)
CAPTION = re.sub(r"[^A-Za-z0-9 !?.-]", "", os.environ.get("CAPTION") or "")
CLIPS = sorted(glob.glob("clips/*.mp4"))
SONGS = glob.glob("songs/*.mp3") + glob.glob("song.mp3")
if not CLIPS or not SONGS:
    raise SystemExit("need clips and a song")
os.makedirs("out", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

def run(cmd):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + cmd, check=True)

def length(path):
    out = subprocess.check_output(["ffprobe", "-v", "error",
        "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(out.decode().strip())

song = random.choice(SONGS)
y, sr = librosa.load(song)
dur = min(DURATION, len(y) / sr)

rms = librosa.feature.rms(y=y, hop_length=sr)[0]
win = int(dur)
offset = 0.0
if len(rms) > win:
    scores = [rms[i:i + win].mean() for i in range(len(rms) - win)]
    offset = float(int(np.argmax(scores)))
seg = y[int(offset * sr):int((offset + dur) * sr)]
print("song:", song, "| start at", offset, "s")

env = librosa.onset.onset_strength(y=seg, sr=sr)
_, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
bt = librosa.frames_to_time(beats, sr=sr)
strength = env[np.minimum(beats, len(env) - 1)]
med = float(np.median(strength))

cuts = [0.0]
skip = False
for t, s in zip(bt, strength):
    if t < 0.4 or t > dur - 0.4:
        continue
    if s >= med:
        cuts.append(float(t))
        skip = False
    else:
        if skip:
            cuts.append(float(t))
        skip = not skip

final = [0.0]
for c in cuts[1:]:
    if c - final[-1] >= 0.25 and dur - c >= 0.25:
        final.append(c)
final.append(dur)
print("segments:", len(final) - 1)

pool, last, lens = [], [None], {}
def pick():
    if not pool:
        pool.extend(CLIPS)
        random.shuffle(pool)
    c = pool.pop()
    if c == last[0] and pool:
        c = pool.pop()
    last[0] = c
    return c

parts = []
for i, (a, b) in enumerate(zip(final, final[1:])):
    length_s = b - a
    clip = pick()
    if clip not in lens:
        lens[clip] = length(clip)
    start = random.uniform(0.3, max(0.3, lens[clip] - length_s - 0.3))
    vf = "crop=ih*9/16:ih,"
    if i % 2:
        vf += "crop=iw/1.15:ih/1.15,"
    vf += "scale=1080:1920,setsar=1,fps=30"
    if i % 4 == 3:
        vf += ",fade=t=in:st=0:d=0.1:color=white"
    out = f"tmp/seg{i}.mp4"
    run(["-ss", f"{start:.2f}", "-i", clip, "-t", f"{length_s:.3f}",
         "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", out])
    parts.append(out)

with open("tmp/list.txt", "w") as f:
    for p in parts:
        f.write("file '" + os.path.basename(p) + "'\n")
run(["-f", "concat", "-safe", "0", "-i", "tmp/list.txt", "-c", "copy", "tmp/video.mp4"])

fade = f"fade=t=out:st={dur - 1:.2f}:d=1"
def mux(vf):
    run(["-i", "tmp/video.mp4", "-ss", f"{offset:.2f}", "-t", f"{dur:.2f}", "-i", song,
         "-map", "0:v", "-map", "1:a", "-vf", vf,
         "-af", f"afade=t=out:st={dur - 1:.2f}:d=1",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "out/edit.mp4"])

if CAPTION:
    text = ("drawtext=text='" + CAPTION + "':fontcolor=white:fontsize=64:box=1:"
            "boxcolor=black@0.5:boxborderw=20:x=(w-text_w)/2:y=h-300:"
            "enable='between(t,0,4)'")
    try:
        mux(text + "," + fade)
    except subprocess.CalledProcessError:
        print("caption failed, rendering without it")
        mux(fade)
else:
    mux(fade)
print("done: out/edit.mp4")
