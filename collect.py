import asyncio, os, re
from playwright.async_api import async_playwright

URL = os.environ.get("ANIME_URL") or "https://animeclips.online/attack-on-titan/"
SEASONS = [s.strip() for s in (os.environ.get("SEASONS") or "Season 1,Season 2,Season 3,Season 4").split(",") if s.strip()]
MAX_CLIPS = int(os.environ.get("MAX_CLIPS") or 12)
PER_EP = int(os.environ.get("PER_EP") or 3)
DELAY = 6000
HAVE = set(open("have.txt").read().split()) if os.path.exists("have.txt") else set()
DONE = set(open("done.txt").read().splitlines()) if os.path.exists("done.txt") else set()

NAMES_JS = """(sel) => [...document.querySelectorAll(sel)].map(
    e => (e.getAttribute('data-name') || e.innerText).trim().split('\\n')[0].trim())"""

def clean(name):
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def spread(items, k):
    if k >= len(items):
        return list(items)
    return [items[int((j + 0.5) * len(items) / k)] for j in range(k)]

def mark_done(key):
    DONE.add(key)
    with open("done.txt", "a") as f:
        f.write(key + "\n")

async def open_season(page, season):
    await page.goto(URL, wait_until="networkidle")
    await page.wait_for_timeout(5000)
    loc = page.get_by_text(season, exact=True)
    for i in range(await loc.count()):
        if await loc.nth(i).is_visible():
            await loc.nth(i).click(timeout=10000)
            break
    await page.wait_for_timeout(8000)

async def main():
    os.makedirs("out", exist_ok=True)
    saved = 0
    fails = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        for season in SEASONS:
            if saved >= MAX_CLIPS or fails >= 3:
                break
            try:
                await open_season(page, season)
            except Exception as e:
                print("cannot open", season, str(e)[:100])
                continue
            episodes = await page.evaluate(NAMES_JS, ".entry.folder[data-id]:not(.pf)")
            print(season, "has", len(episodes), "folders")
            for idx, ep in enumerate(episodes):
                if saved >= MAX_CLIPS or fails >= 3:
                    break
                key = f"{URL}|{season}|{ep}|{PER_EP}"
                if key in DONE:
                    continue
                try:
                    await open_season(page, season)
                    await page.locator(".entry.folder[data-id]:not(.pf)").nth(idx).click(timeout=10000)
                    await page.wait_for_timeout(8000)
                    names = await page.evaluate(NAMES_JS, ".entry.file[data-id]")
                except Exception as e:
                    print("cannot open", ep, str(e)[:100])
                    continue
                have_here = [i for i, n in enumerate(names) if n and clean(n) in HAVE]
                todo = [i for i, n in enumerate(names) if n and clean(n) not in HAVE]
                need = PER_EP - len(have_here)
                print(season, ep, ":", len(names), "clips,", len(have_here), "owned, need", max(0, need))
                if need <= 0 or not todo:
                    mark_done(key)
                    continue
                chosen = spread(todo, min(need, MAX_CLIPS - saved))
                files = page.locator(".entry.file[data-id]")
                all_ok = True
                for i in chosen:
                    if fails >= 3:
                        all_ok = False
                        break
                    try:
                        entry = files.nth(i)
                        await entry.click(timeout=8000)
                        await page.wait_for_timeout(3000)
                        async with page.expect_download(timeout=60000) as dl:
                            await entry.locator("a.entry_action_download").first.evaluate("e => e.click()")
                        d = await dl.value
                        name = clean(d.suggested_filename)
                        await d.save_as(os.path.join("out", name))
                        HAVE.add(name)
                        saved += 1
                        fails = 0
                        print("saved", name)
                    except Exception as e:
                        fails += 1
                        all_ok = False
                        print("clip failed:", str(e)[:120].replace("\n", " "))
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(DELAY)
                if all_ok and len(chosen) == need:
                    mark_done(key)
        print("done:", saved, "saved")
        await browser.close()

asyncio.run(main())
