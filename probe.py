import asyncio, os
from playwright.async_api import async_playwright

URL = "https://animeclips.online/attack-on-titan/"
FILE = ".entry.file[data-id]"

async def try_download(page, label, action):
    try:
        async with page.expect_download(timeout=25000) as dl:
            await action()
        d = await dl.value
        path = os.path.join("out", d.suggested_filename)
        await d.save_as(path)
        print("SUCCESS via", label, "|", d.suggested_filename, os.path.getsize(path), "bytes")
        return True
    except Exception as e:
        print("failed via", label, "|", str(e)[:120].replace("\n", " "))
        return False

async def main():
    os.makedirs("out", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(5000)
        await page.get_by_text("Season 1", exact=True).nth(1).click(timeout=8000)
        await page.wait_for_timeout(8000)
        await page.locator(".entry[data-id]:not(.pf)").first.click(timeout=8000)
        await page.wait_for_timeout(8000)

        entry = page.locator(FILE).first
        await entry.click(timeout=8000)
        await page.wait_for_timeout(6000)

        print("=== DOWNLOAD BUTTONS ===")
        info = await page.evaluate("""() => [...document.querySelectorAll('a.entry_action_download, .eva-download')].map(e => {
            const r = e.getBoundingClientRect();
            const par = e.parentElement ? e.parentElement.className.slice(0, 40) : '';
            return e.tagName + ' | ' + e.className.slice(0, 40) + ' | parent: ' + par + ' | size ' + Math.round(r.width) + 'x' + Math.round(r.height);
        })""")
        for i in info:
            print(i)

        ok = await try_download(page, "visible preview button",
            lambda: page.locator("a.entry_action_download:visible").first.click(timeout=5000))
        if not ok:
            ok = await try_download(page, "js click",
                lambda: page.evaluate("() => document.querySelector('a.entry_action_download').click()"))
        if not ok:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(2000)
            await entry.hover()
            ok = await try_download(page, "hover then click",
                lambda: entry.locator("a.entry_action_download").first.click(timeout=5000, force=True))
        await browser.close()

asyncio.run(main())
