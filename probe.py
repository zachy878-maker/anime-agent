import asyncio
from playwright.async_api import async_playwright

URL = "https://animeclips.online/attack-on-titan/"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(5000)

        def log_req(r):
            if "admin-ajax" in r.url:
                print("AJAX:", (r.post_data or "")[:400])
        page.on("request", log_req)

        await page.get_by_text("Season 1", exact=True).first.click()
        await page.wait_for_timeout(10000)

        print("=== FILE ROWS ===")
        rows = await page.evaluate("""() => [...document.querySelectorAll('[class*="entry"]')]
            .slice(0, 6).map(e => e.className + ' | ' + e.innerText.slice(0, 80))""")
        for r in rows:
            print(r)
        print("=== TURNSTILE ===")
        print(await page.locator("iframe[src*='turnstile']").count(), "turnstile frames")
        print("=== PAGE TEXT ===")
        print((await page.inner_text("body"))[:1500])
        await browser.close()

asyncio.run(main())
