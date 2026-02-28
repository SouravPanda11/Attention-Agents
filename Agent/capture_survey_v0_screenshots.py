import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright


PAGE_SEQUENCE = (
    ("text", "button:has-text('Next')"),
    ("image", "button:has-text('Next')"),
    ("thank-you", "button:has-text('Submit Responses')"),
    ("done", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture full-page screenshots for survey_v0 only "
            "by following a fixed page flow (no question answering)."
        )
    )
    parser.add_argument(
        "--start-url",
        default="http://localhost:3000/survey",
        help="Survey root URL. Must point to survey_v0 route (/survey).",
    )
    parser.add_argument(
        "--output-dir",
        default="Agent/runs/survey_v0/manual_fullpage_screenshots",
        help="Directory where screenshot run folders are created.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Navigation/click timeout in milliseconds.",
    )
    return parser.parse_args()


def ensure_v0_route(start_url: str) -> str:
    parsed = urlparse(start_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/survey_v1") or "/survey_v1/" in f"{path}/":
        raise ValueError("Only survey_v0 is supported. Use /survey route, not /survey_v1.")
    if not path.endswith("/survey"):
        raise ValueError("Only survey_v0 is supported. Start URL must end with /survey.")
    return f"{parsed.scheme}://{parsed.netloc}{path}/"


def build_run_output_dir(base_output_dir: str) -> Path:
    root = Path(base_output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


async def wait_for_route(page, route_name: str, timeout_ms: int) -> None:
    await page.wait_for_url(
        lambda url: urlparse(url).path.rstrip("/").endswith(f"/survey/{route_name}"),
        timeout=timeout_ms,
    )
    await page.wait_for_load_state("networkidle")


async def capture_fullpage(page, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    await page.screenshot(path=str(out_path), full_page=True)


def save_pngs_as_pdf(png_paths: list[Path], pdf_path: Path) -> None:
    if not png_paths:
        raise ValueError("No PNG screenshots were captured; cannot create PDF.")

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required to create PDF. Install with: pip install pillow"
        ) from exc

    pages = []
    for png_path in png_paths:
        with Image.open(png_path) as image:
            pages.append(image.convert("RGB"))

    first_page, *rest_pages = pages
    try:
        first_page.save(str(pdf_path), "PDF", save_all=True, append_images=rest_pages)
    finally:
        for page in pages:
            page.close()


async def run() -> int:
    args = parse_args()
    survey_root = ensure_v0_route(args.start_url)
    output_dir = build_run_output_dir(args.output_dir)

    print(f"[capture] survey_root={survey_root}")
    print(f"[capture] output_dir={output_dir}")

    screenshot_paths: list[Path] = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=args.headless)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(survey_root, wait_until="networkidle", timeout=args.timeout_ms)

            for idx, (route_name, action_selector) in enumerate(PAGE_SEQUENCE, start=1):
                await wait_for_route(page, route_name, args.timeout_ms)
                shot_path = output_dir / f"{idx:02d}_{route_name}.png"
                await capture_fullpage(page, shot_path)
                screenshot_paths.append(shot_path)
                print(f"[capture] saved={shot_path}")

                if action_selector is None:
                    continue

                await page.locator(action_selector).first.click(timeout=args.timeout_ms)
                await page.wait_for_load_state("networkidle")
        finally:
            await context.close()
            await browser.close()

    pdf_path = output_dir / "all_pages.pdf"
    save_pngs_as_pdf(screenshot_paths, pdf_path)
    print(f"[capture] saved={pdf_path}")

    print("[capture] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
