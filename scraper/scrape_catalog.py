"""Optional scraper for refreshing data/catalog.json.

The deployed application does not import or run this module during startup. When
the output catalog already exists, the command exits without making live SHL
requests unless --force is supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.models import Assessment
from app.utils import load_json, write_json

CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = BASE_DIR / "data" / "catalog.json"
logger = logging.getLogger(__name__)


def normalize_bool(value: str) -> bool:
    """Convert SHL table icon/label text into a boolean."""

    return value.strip().lower() in {"yes", "true", "y", "available", "✓", "check"}


def clean_text(value: str) -> str:
    """Collapse repeated whitespace."""

    return re.sub(r"\s+", " ", value).strip()


async def scrape_with_playwright() -> list[Assessment]:
    """Scrape the live SHL catalog with Playwright."""

    from playwright.async_api import async_playwright

    assessments: list[Assessment] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(CATALOG_URL, wait_until="networkidle", timeout=60000)
        await _select_individual_test_solutions(page)
        rows = await page.locator("table tbody tr").all()
        for row in rows:
            assessment = await _parse_playwright_row(row)
            if assessment:
                assessments.append(assessment)
        await browser.close()
    return _deduplicate(assessments)


async def _select_individual_test_solutions(page) -> None:
    """Filter the catalog to Individual Test Solutions when controls exist."""

    candidates = [
        "text=Individual Test Solutions",
        "label:has-text('Individual Test Solutions')",
        "[aria-label*='Individual Test Solutions']",
    ]
    for selector in candidates:
        try:
            control = page.locator(selector).first
            if await control.count():
                await control.click(timeout=5000)
                await page.wait_for_load_state("networkidle")
                return
        except Exception:
            continue


async def _parse_playwright_row(row) -> Assessment | None:
    cells = [clean_text(text) for text in await row.locator("td").all_inner_texts()]
    if not cells:
        return None
    link = row.locator("a").first
    href = await link.get_attribute("href") if await link.count() else ""
    name = cells[0]
    if not name or "job solution" in " ".join(cells).lower():
        return None
    return Assessment(
        name=name,
        description=cells[1] if len(cells) > 1 else name,
        skills_measured=_split_list(cells[2] if len(cells) > 2 else name),
        test_type=cells[3] if len(cells) > 3 else "Assessment",
        duration=cells[4] if len(cells) > 4 else "See SHL catalog",
        remote_testing_support=normalize_bool(cells[5]) if len(cells) > 5 else True,
        adaptive_support=normalize_bool(cells[6]) if len(cells) > 6 else False,
        job_levels=_split_list(cells[7] if len(cells) > 7 else "All levels"),
        languages=_split_list(cells[8] if len(cells) > 8 else "English"),
        url=urljoin(CATALOG_URL, href),
    )


def scrape_with_beautifulsoup() -> list[Assessment]:
    """Best-effort non-JavaScript scrape of visible catalog markup."""

    import requests
    from bs4 import BeautifulSoup

    response = requests.get(CATALOG_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    assessments: list[Assessment] = []
    for row in soup.select("table tbody tr"):
        cells = [clean_text(cell.get_text(" ")) for cell in row.select("td")]
        if not cells:
            continue
        link = row.select_one("a[href]")
        href = link["href"] if link else ""
        if "job solution" in " ".join(cells).lower():
            continue
        assessments.append(
            Assessment(
                name=cells[0],
                description=cells[1] if len(cells) > 1 else cells[0],
                skills_measured=_split_list(cells[2] if len(cells) > 2 else cells[0]),
                test_type=cells[3] if len(cells) > 3 else "Assessment",
                duration=cells[4] if len(cells) > 4 else "See SHL catalog",
                remote_testing_support=normalize_bool(cells[5]) if len(cells) > 5 else True,
                adaptive_support=normalize_bool(cells[6]) if len(cells) > 6 else False,
                job_levels=_split_list(cells[7] if len(cells) > 7 else "All levels"),
                languages=_split_list(cells[8] if len(cells) > 8 else "English"),
                url=urljoin(CATALOG_URL, href),
            )
        )
    return _deduplicate(assessments)


def _split_list(value: str) -> list[str]:
    parts = re.split(r"[,;/|]", value)
    return [clean_text(part) for part in parts if clean_text(part)]


def _deduplicate(items: list[Assessment]) -> list[Assessment]:
    by_url: dict[str, Assessment] = {}
    for item in items:
        by_url[item.url or item.name] = item
    return sorted(by_url.values(), key=lambda assessment: assessment.name.lower())


async def scrape_catalog() -> list[Assessment]:
    """Scrape catalog using the best available engine.

    This function is intentionally only used by the manual scraper command. The
    FastAPI app loads data/catalog.json directly and never calls live SHL pages.
    """

    try:
        assessments = await scrape_with_playwright()
        if assessments:
            return assessments
    except Exception as exc:
        logger.warning("Playwright scrape failed: %s", exc)

    try:
        assessments = scrape_with_beautifulsoup()
        if assessments:
            return assessments
    except Exception as exc:
        logger.warning("BeautifulSoup scrape failed: %s", exc)

    raise RuntimeError("Unable to scrape SHL catalog. Check network access and site markup.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Scrape SHL and overwrite the output file even when it already exists.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.output.exists() and not args.force:
        raw = load_json(args.output)
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(f"Existing catalog is empty or invalid: {args.output}")
        logger.info("catalog already exists at %s; skipping live scraping", args.output)
        return

    assessments = asyncio.run(scrape_catalog())
    write_json(args.output, [assessment.model_dump() for assessment in assessments])
    logger.info("wrote %s assessments to %s", len(assessments), args.output)


if __name__ == "__main__":
    main()
