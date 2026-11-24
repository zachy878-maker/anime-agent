import asyncio
from playwright.async_api import async_playwright

URL = "https://animeclips.online/attack-on-titan/"

async def show(page, label):
    items = await page.evaluate("""() => [...document.querySelectorAll('.entry[data-id]')]
        .map(e => e.className.slice(0, 40) + ' | ' + (e.getAttribute('data-name') || e.innerText.slice(0, 50)))""")
    print(f"=== {label}: {len(items)} entries ===")
    for i in items[:8]:
        print(i)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(5000)

        try:
            await page.get_by_text("Season 1", exact=True).nth(1).click(timeout=8000)
            await page.wait_for_timeout(8000)
            await show(page, "SEASON 1 CONTENTS")

            await page.locator(".entry[data-id]").first.click(timeout=8000)
            await page.wait_for_timeout(8000)
            await show(page, "FIRST EPISODE CONTENTS")

            await page.locator(".entry[data-id]").first.click(timeout=8000)
            await page.wait_for_timeout(8000)
            print("=== AFTER CLICKING FIRST CLIP ===")
            found = await page.evaluate("""() => {
                const out = [];
                document.querySelectorAll('a[download], video, source, [class*="download"], [data-action*="download"]').forEach(e => {
                    out.push(e.tagName + ' | ' + e.className + ' | ' + (e.getAttribute('data-action') || '') + ' | ' + (e.href || e.src || '').slice(0, 150));
                });
                return out.slice(0, 15);
            }""")
            for f in found:
                print(f)
        except Exception as e:
            print("STOPPED:", str(e)[:400])

        await browser.close()

asyncio.run(main())
