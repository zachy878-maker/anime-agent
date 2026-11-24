import asyncio
from playwright.async_api import async_playwright

URL = "https://animeclips.online/attack-on-titan/"
SKIP = (".png", ".jpg", ".jpeg", ".svg", ".css", ".woff", ".woff2", ".gif", ".webp")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        reqs = []
        page.on("request", lambda r: reqs.append(f"{r.method} {r.url}"))
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(10000)
        print("=== REQUESTS ===")
        for r in reqs:
            if not r.split("?")[0].lower().endswith(SKIP):
                print(r[:200])
        print("=== PAGE TEXT ===")
        print((await page.inner_text("body"))[:2500])
        await browser.close()

asyncio.run(main())
