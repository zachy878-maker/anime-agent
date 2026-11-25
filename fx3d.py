import math, random, subprocess
import numpy as np
import cv2

IDS = ("depth-anything/Depth-Anything-V2-Small-hf", "LiheYoung/depth-anything-small-hf",
       "Intel/dpt-hybrid-midas")
_pipe = None

def depth_of(rgb):
    global _pipe
    from PIL import Image
    if _pipe is None:
        from transformers import pipeline
        err = None
        for name in IDS:
            try:
                _pipe = pipeline("depth-estimation", model=name)
                break
            except Exception as e:
                err = e
        if _pipe is None:
            raise err
    d = _pipe(Image.fromarray(rgb))["depth"]
    d = np.asarray(d.resize((rgb.shape[1], rgb.shape[0])), dtype=np.float32)
    d -= d.min()
    return d / (d.max() + 1e-6)

KINDS = {"petals": ((205, 175, 255), 3.0, 1.0, "fall"),
         "sparks": ((60, 160, 255), 1.5, 1.0, "rise"),
         "dust": ((230, 230, 230), 1.2, 0.3, "fall")}
LIGHT = {"petals": (110, 150, 255), "sparks": (60, 140, 255), "dust": (200, 215, 255)}

def hero(png, nfr, out, W, H, fps, kind):
    kind = kind if kind in KINDS else "dust"
    img = cv2.resize(cv2.imread(png), (W, H))
    depth = depth_of(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    depth = cv2.GaussianBlur(depth, (0, 0), 3)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
    nx, ny = -gx * 6.0, -gy * 6.0
    nrm = np.sqrt(nx * nx + ny * ny + 1.0)
    nx, ny, nz = nx / nrm, ny / nrm, 1.0 / nrm
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    amp = 0.022 * W
    color, rad, spd, mode = KINDS[kind]
    light_vec = np.array(LIGHT[kind], np.float32) / 255 * 150
    rnd = random.Random(11)
    ps = [{"x": rnd.uniform(0, W), "y": rnd.uniform(0, H), "z": rnd.random(),
           "ph": rnd.uniform(0, 6.28)} for _ in range(110)]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for k in range(nfr):
        t = k / max(1, nfr - 1)
        dxs = (t - 0.5) * 2 * amp
        zoom = 1.04 + 0.05 * t
        sh = (depth - 0.5) * dxs
        mx = ((xx - W / 2) / zoom + W / 2 + sh).astype(np.float32)
        my = ((yy - H / 2) / zoom + H / 2 + (depth - 0.5) * dxs * 0.25).astype(np.float32)
        frame = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        dep = cv2.remap(depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        lx, ly, lz = W * (0.15 + 0.7 * t), H * 0.28, 0.35 * W
        lxv, lyv, lzv = lx - xx, ly - yy, lz - dep * 0.12 * W
        ln = np.sqrt(lxv ** 2 + lyv ** 2 + lzv ** 2) + 1e-6
        lam = np.clip((nx * lxv + ny * lyv + nz * lzv) / ln, 0, 1)
        fall = 1.0 / (1.0 + (ln / (0.55 * W)) ** 2)
        frame += (lam * fall)[..., None] * light_vec
        ov = np.zeros((H, W, 3), np.float32)
        for p in ps:
            v = (1.2 + p["z"] * 3.2) * spd * H / 540
            p["y"] += v if mode == "fall" else -v
            p["x"] = (p["x"] + math.sin(k * 0.05 + p["ph"]) * 1.4 + (0.8 if mode == "fall" else 0.3)) % W
            if p["y"] > H + 10:
                p["y"] = -10
            if p["y"] < -10:
                p["y"] = H + 10
            xi, yi = int(p["x"]), int(p["y"])
            if not (0 <= xi < W and 0 <= yi < H) or dep[yi, xi] > p["z"] + 0.03:
                continue
            r = max(1, int((rad + p["z"] * 5) * H / 1080))
            cv2.circle(ov, (xi, yi), r, tuple(float(c) for c in color), -1, lineType=cv2.LINE_AA)
        ov = cv2.GaussianBlur(ov, (0, 0), 1.2)
        frame += ov * 0.85
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("parallax encode failed")
