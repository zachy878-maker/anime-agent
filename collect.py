import asyncio, os, re
from playwright.async_api import async_playwright

URL = "https://animeclips.online/attack-on-titan/"
SEASON = os.environ.get("SEASON", "Season 1")
EPISODE = os.environ.get("EPISODE", "1")
COUNT = int(os.environ.get("COUNT", "3"))

async def main():
    os.makedirs("out", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(5000)
        await page.get_by_text(SEASON, exact=True).nth(1).click(timeout=8000)
        await page.wait_for_timeout(8000)
        folder = page.locator(".entry.folder[data-id]:not(.pf)").filter(
            has_text=re.compile(rf"Episode {EPISODE}\b"))
        await folder.first.click(timeout=8000)
        await page.wait_for_timeout(8000)
        files = page.locator(".entry.file[data-id]")
        total = await files.count()
        print("clips in folder:", total)
        saved = 0
        for i in range(min(COUNT, total)):
            try:
                entry = files.nth(i)
                await entry.click(timeout=8000)
                await page.wait_for_timeout(3000)
                async with page.expect_download(timeout=60000) as dl:
                    await entry.locator("a.entry_action_download").first.evaluate("e => e.click()")
                d = await dl.value
                name = re.sub(r"[^A-Za-z0-9._-]+", "_", d.suggested_filename)
                path = os.path.join("out", name)
                await d.save_as(path)
                print("saved", name, os.path.getsize(path), "bytes")
                saved += 1
            except Exception as e:
                print("clip", i + 1, "failed:", str(e)[:150].replace("\n", " "))
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(6000)
        print("done:", saved, "saved")
        await browser.close()

asyncio.run(main())
