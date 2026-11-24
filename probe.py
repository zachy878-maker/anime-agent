import asyncio, os
from playwright.async_api import async_playwright

URL = "https://animeclips.online/attack-on-titan/"
REAL = ".entry[data-id]:not(.pf)"

async def show(page, label):
    items = await page.evaluate("""() => [...document.querySelectorAll('.entry[data-id]:not(.pf)')]
        .map(e => e.className.slice(0, 40) + ' | ' + (e.getAttribute('data-name') || e.innerText.slice(0, 50)))""")
    print(f"=== {label}: {len(items)} entries ===")
    for i in items[:8]:
        print(i)

async def main():
    os.makedirs("out", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(5000)
        try:
            await page.get_by_text("Season 1", exact=True).nth(1).click(timeout=8000)
            await page.wait_for_timeout(8000)
            await page.locator(REAL).first.click(timeout=8000)
            await page.wait_for_timeout(8000)
            await show(page, "EPISODE 1 CONTENTS")

            await page.locator(".entry[data-id]:not(.pf):not(.folder)").first.click(timeout=8000)
            await page.wait_for_timeout(8000)
            print("=== TRYING ONE DOWNLOAD ===")
            async with page.expect_download(timeout=60000) as dl:
                await page.locator("a.entry_action_download:visible").first.click()
            d = await dl.value
            path = os.path.join("out", d.suggested_filename)
            await d.save_as(path)
            print("FILE:", d.suggested_filename, os.path.getsize(path), "bytes")
        except Exception as e:
            print("STOPPED:", str(e)[:500])
        await browser.close()

asyncio.run(main())
