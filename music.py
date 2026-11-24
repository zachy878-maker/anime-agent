import os, re, subprocess, time
import requests

QUERY = os.environ.get("QUERY") or "electronic"
MAX_TRACKS = int(os.environ.get("MAX_TRACKS") or 10)
API = "https://api.openverse.org/v1/audio/"
HEADERS = {"User-Agent": "anime-agent-music/1.0"}
HAVE = set(open("have_songs.txt").read().split()) if os.path.exists("have_songs.txt") else set()

def clean(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:60]

def search(page):
    r = requests.get(API, headers=HEADERS, timeout=60, params={
        "q": QUERY, "category": "music",
        "license_type": "commercial,modification",
        "page_size": 20, "page": page})
    if r.status_code == 429:
        raise SystemExit("rate limited, try again later")
    r.raise_for_status()
    return r.json().get("results", [])

os.makedirs("songs_out", exist_ok=True)
saved = 0
for page in range(1, 4):
    if saved >= MAX_TRACKS:
        break
    results = search(page)
    if not results:
        break
    for t in results:
        if saved >= MAX_TRACKS:
            break
        dur = (t.get("duration") or 0) / 1000
        if dur < 60 or dur > 420 or not t.get("url"):
            continue
        name = f"{clean(t.get('title') or 'track')}_{t['id'][:8]}.mp3"
        if name in HAVE:
            continue
        try:
            raw = "songs_out/raw"
            with requests.get(t["url"], headers=HEADERS, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(raw, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                            "-vn", "-ac", "2", "-b:a", "192k",
                            os.path.join("songs_out", name)], check=True)
            os.remove(raw)
            lic = f"{t.get('license')} {t.get('license_version') or ''}".strip()
            with open("credits.csv", "a") as f:
                f.write("|".join([name, t.get("title") or "", t.get("creator") or "",
                                  lic, t.get("foreign_landing_url") or ""]) + "\n")
            HAVE.add(name)
            saved += 1
            print("saved", name, "|", lic, "|", t.get("creator"))
        except Exception as e:
            print("failed:", str(e)[:120].replace("\n", " "))
        time.sleep(3)
    time.sleep(4)
print("done:", saved, "songs")
