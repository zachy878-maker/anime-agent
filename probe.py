import asyncio
from playwright.async_api import async_playwright

URL = "https://animeclips.online/attack-on-titan/"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(5000)

        loc = page.get_by_text("Season 1", exact=True)
        try:
            await loc.nth(1).click(timeout=8000)
            await page.wait_for_timeout(10000)
            print("clicked ok")
        except Exception as e:
            print("CLICK FAILED:", str(e)[:300])

        count = await page.evaluate("() => document.querySelectorAll('.entry[data-id]').length")
        print("=== ENTRY COUNT ===", count)

        names = await page.evaluate("""() => [...document.querySelectorAll('.entry[data-id]')]
            .slice(0, 10).map(e => e.getAttribute('data-name') || e.innerText.slice(0, 60))""")
        print("=== FIRST NAMES ===")
        for n in names:
            print(n)

        html = await page.evaluate("""() => {
            const e = document.querySelector('.entry[data-id]');
            return e ? e.outerHTML.slice(0, 900) : 'none';
        }""")
        print("=== FIRST ENTRY HTML ===")
        print(html)

        btns = await page.evaluate("""() => [...document.querySelectorAll('[class*="download"], [data-action*="download"]')]
            .slice(0, 8).map(e => e.tagName + ' ' + e.className + ' ' + (e.getAttribute('data-action') || ''))""")
        print("=== DOWNLOAD BUTTONS ===")
        for b in btns:
            print(b)
        await browser.close()

asyncio.run(main())
