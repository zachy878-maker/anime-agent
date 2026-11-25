import json, os, re, subprocess, sys
import numpy as np

REF = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "custom"
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", REF]).decode().strip())
r = subprocess.run(["ffmpeg", "-i", REF, "-vf", "select='gt(scene,0.28)',showinfo", "-an", "-f", "null", "-"],
                   capture_output=True, text=True)
times = [float(x) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]
edges = [0.0] + times + [dur]
lens = [b - a for a, b in zip(edges, edges[1:]) if b - a > 0.15]

out = subprocess.run(["ffmpeg", "-v", "error", "-i", REF, "-vf",
                      "fps=2,scale=160:-2,signalstats,metadata=mode=print:file=-", "-an", "-f", "null", "-"],
                     capture_output=True, text=True).stdout
vals = {"SATAVG": [], "UAVG": [], "VAVG": []}
for line in out.splitlines():
    if "lavfi.signalstats." in line:
        k, v = line.split("=", 1)
        k = k.split(".")[-1]
        if k in vals:
            try:
                vals[k].append(float(v))
            except ValueError:
                pass
sat = float(np.mean(vals["SATAVG"])) if vals["SATAVG"] else 35.0
warm = float(np.mean(vals["VAVG"]) - np.mean(vals["UAVG"])) if vals["VAVG"] else 0.0

style = {}
if len(lens) >= 3:
    q = lambda p: float(np.clip(np.percentile(lens, p), 0.5, 6.0))
    style.update({"lo": round(q(75), 2), "mid": round(q(50), 2), "hi": round(q(25), 2)})
    style["xf"] = 6 if q(50) < 1.2 else (10 if q(50) < 2.5 else 12)
    style["score"] = "hype" if q(50) < 1.3 else "romance"
else:
    print("fewer than 3 shots found, keeping the default pacing")
style["sat"] = round(float(np.clip(0.85 + sat / 150, 0.85, 1.3)), 3)
style["warm"] = round(float(np.clip(warm / 600, -0.05, 0.06)), 3)
os.makedirs("styles", exist_ok=True)
json.dump(style, open(f"styles/{NAME}.json", "w"), indent=1)
print("shots:", len(lens), "| saved style:", style)
