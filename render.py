import glob, os, random, subprocess
import librosa

DURATION = 20
CLIPS = sorted(glob.glob("clips/*.mp4"))
if not CLIPS:
    raise SystemExit("no clips found")
os.makedirs("out", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

def length(path):
    out = subprocess.check_output(["ffprobe", "-v", "error",
        "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(out.decode().strip())

def run(cmd):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + cmd, check=True)

y, sr = librosa.load("song.mp3", duration=DURATION)
end = len(y) / sr
_, beats = librosa.beat.beat_track(y=y, sr=sr)
times = [float(t) for t in librosa.frames_to_time(beats, sr=sr)][::2]
cuts = [0.0] + [t for t in times if 0.3 < t < end - 0.5] + [end]
print("cuts:", len(cuts) - 1, "segments, song length", round(end, 1))

parts = []
for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
    seg = b - a
    if seg < 0.2:
        continue
    clip = random.choice(CLIPS)
    start = random.uniform(0, max(0, length(clip) - seg))
    out = f"tmp/seg{i}.mp4"
    run(["-ss", f"{start:.2f}", "-i", clip, "-t", f"{seg:.3f}",
         "-vf", "crop=ih*9/16:ih,scale=1080:1920,setsar=1,fps=30",
         "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", out])
    parts.append(out)

with open("tmp/list.txt", "w") as f:
    for p in parts:
        f.write("file '" + os.path.basename(p) + "'\n")

run(["-f", "concat", "-safe", "0", "-i", "tmp/list.txt", "-c", "copy", "tmp/video.mp4"])
run(["-i", "tmp/video.mp4", "-i", "song.mp3", "-map", "0:v", "-map", "1:a",
     "-c:v", "copy", "-c:a", "aac", "-shortest", "out/edit.mp4"])
print("done: out/edit.mp4")
