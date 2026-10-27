#!/usr/bin/env python3
"""Collect custom GDDL demons, showcase videos, and difficulty vote histograms.

The site is a JavaScript application, so this scraper uses a real browser and
reads the rendered pages. It does not call GDDL's data endpoints directly.

Requires: pip install playwright && playwright install chromium

The output is CSV. ``difficulty_vote_distribution`` is a Python dictionary
literal with integer tier keys and integer vote counts. Showcase links are JSON.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

BASE_URL = "https://gdladder.com"
PAGE_RETRY_DELAY_SECONDS = 60


class PageLoadError(RuntimeError):
    """Raised when navigation itself fails, rather than page data being absent."""


class LevelDataNotRendered(RuntimeError):
    """Raised when a loaded level page does not render its required fields."""


def is_retryable_http_error(response) -> bool:
    if response is None:
        return False
    try:
        return response.status in {403, 429, 500, 502, 503, 504}
    except Exception:
        return False


def is_browser_error_page(page) -> bool:
    """Detect browser/network error screens; missing site labels are not this."""
    current_url = (page.url or "").lower()
    if current_url.startswith(("chrome-error://", "about:neterror")):
        return True
    try:
        body = page.locator("body").inner_text(timeout=1000).lower()
    except Exception:
        return False
    return any(marker in body for marker in (
        "this site can’t be reached",
        "this site can't be reached",
        "webpage not available",
        "err_connection_reset",
        "err_connection_refused",
        "err_name_not_resolved",
        "err_timed_out",
        "problem loading page",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
    ))


def is_youtube_video_url(url: str) -> bool:
    """Accept direct video URLs only; exclude search, channel, and playlist URLs."""
    from urllib.parse import parse_qs

    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":", 1)[0]
    path = parsed.path.strip("/")
    if host == "youtu.be":
        return bool(path and path.split("/")[0])
    if host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"}:
        return False
    if path == "watch":
        return bool(parse_qs(parsed.query).get("v", [""])[0])
    if path.startswith(("shorts/", "live/", "embed/", "v/")):
        return bool(path.split("/", 1)[1])
    return False


def rendered_level_length(page) -> str:
    """Read the displayed length; map Platformer and XL+ categories explicitly."""
    text = page.locator("body").inner_text(timeout=1500)
    match = re.search(r"\bLength\s*:?\s*(Platformer|Tiny|Short|Medium|Long|(?:\d+)?XL|X{2,}L)\b", text, re.I)
    if not match:
        return ""
    return normalize_length_label(match.group(1))


def normalize_length_label(value: str) -> str:
    value = value.strip().lower()
    if value == "platformer":
        return "Platformer"
    if re.fullmatch(r"(?:\d+)?xl|x{2,}l", value):
        return "XL"
    if value in {"tiny", "short", "medium", "long"}:
        return value.capitalize()
    return ""


def wait_for_level_length(page, level_id: str, timeout_ms: int = 8000) -> str:
    """Wait for the level's metadata card, which can hydrate after navigation."""
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        try:
            length = rendered_level_length(page)
            if length:
                return length
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise LevelDataNotRendered(f"Length label did not render for level {level_id}")
        page.wait_for_timeout(200)


def level_pager_state(page) -> dict:
    """Read the level-list pager, excluding the separate submission pager."""
    matches = page.locator("button, a").evaluate_all("""
          els => els.map((el, index) => {
            if (el.getAttribute('aria-label') !== 'Next page') return null;
            let node = el;
            for (let depth = 0; node && depth < 8; depth++, node = node.parentElement) {
              const text = (node.innerText || '').replace(/\\s+/g, ' ').trim();
              // The level list's pager is rendered in a `my-4` wrapper.
              // Submission results have their own Next button inside a
              // section, so require that wrapper to avoid paging submissions.
              if (node.classList?.contains('my-4') && node.querySelector('input') &&
                  /\\bPage\\s*\\/\\s*[\\d,]+/i.test(text)) {
                const totalMatch = text.match(/\\bPage\\s*\\/\\s*([\\d,]+)/i);
                const input = node.querySelector('input');
                return {index, current: node.querySelector('input')?.value || '',
                  inputIndex: [...document.querySelectorAll('input')].indexOf(input),
                  total: totalMatch ? totalMatch[1].replace(/,/g, '') : '',
                  disabled: el.disabled || el.getAttribute('aria-disabled') === 'true'};
              }
            }
            return null;
          }).filter(Boolean).slice(0, 1)
    """)
    if not matches:
        raise RuntimeError("Could not find the demon-list pagination control")
    return matches[0]


def go_to_list_page(page, target_page: int) -> None:
    """Use the rendered page-number input to resume at a specific list page."""
    state = level_pager_state(page)
    current = int(state["current"]) if str(state["current"]).isdigit() else 0
    if current == target_page:
        return
    if state["inputIndex"] < 0:
        raise RuntimeError("Could not find the demon-list page-number input")
    before = rendered_page_level_ids(page)
    field = page.locator("input").nth(state["inputIndex"])
    field.fill(str(target_page))
    field.press("Enter")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        page.wait_for_timeout(250)
        updated = level_pager_state(page)
        after = rendered_page_level_ids(page)
        if updated["current"] == str(target_page) and after and after != before:
            return
    raise RuntimeError(f"Could not navigate directly to demon-list page {target_page}")


def recover_list_page(page, start_url: str, target_page: int) -> None:
    """Reload the list and restore its page cursor after a transient failure."""
    print(f"Reloading the search page and resuming at list page {target_page}", file=sys.stderr)
    load_rendered_list(page, start_url)
    if target_page > 1:
        go_to_list_page(page, target_page)


def click_next_page(page) -> bool:
    """Click the level-list pager's next control and confirm cards changed."""
    before = rendered_page_level_ids(page)

    state = level_pager_state(page)
    current_page = int(state["current"]) if str(state["current"]).isdigit() else 0
    total_pages = int(state["total"]) if str(state["total"]).isdigit() else 0
    if current_page and total_pages and current_page >= total_pages:
        return False
    if state["disabled"]:
        if not current_page or not total_pages:
            raise RuntimeError("Next page is disabled but pager page numbers could not be read")
        raise RuntimeError(f"Next page is disabled unexpectedly at {current_page}/{total_pages}")

    control = page.locator("button, a").nth(state["index"])
    if not control.is_visible() or not control.is_enabled():
        raise RuntimeError("Demon-list Next page control is temporarily unavailable")
    control.click(timeout=2000)
    deadline = time.monotonic() + 12
    page_number_changed = False
    while time.monotonic() < deadline:
        page.wait_for_timeout(250)
        after = rendered_page_level_ids(page)
        if after and after != before:
            return True
        try:
            updated = level_pager_state(page)
            page_number_changed = updated["current"] != state["current"]
        except Exception:
            pass
        if page_number_changed:
            # The UI accepted the page change; allow slow cards to hydrate
            # without clicking again and accidentally skipping a page.
            cards_deadline = time.monotonic() + 20
            while time.monotonic() < cards_deadline:
                page.wait_for_timeout(250)
                after = rendered_page_level_ids(page)
                if after and after != before:
                    return True
            raise RuntimeError("Paginator changed its page number but level cards did not refresh")
    raise RuntimeError("Demon-list pager did not render a new page")


def sort_list_by_rating_count(page) -> None:
    """Restore the intended order after reloading the list page."""
    try:
        page.get_by_role("button", name=re.compile(r"^Sort by:")).click(timeout=3000)
        page.get_by_role("button", name="Rating count", exact=True).click(timeout=3000)
        direction_button = page.get_by_role("button", name=re.compile(r"Sort ascending|Sort descending"))
        if direction_button.count() and direction_button.get_attribute("aria-label") == "Sort ascending":
            direction_button.click(timeout=3000)
        page.wait_for_timeout(800)
    except Exception:
        # Sorting is a convenience; pagination recovery must still be able to
        # proceed if the sort menu is absent or the saved sort was retained.
        pass


def load_rendered_list(page, start_url: str) -> None:
    """Load the list; use the long cooldown only when navigation itself fails."""
    while True:
        try:
            response = page.goto(start_url, wait_until="domcontentloaded", timeout=20000)
        except Exception as error:
            print(f"Search-page navigation failed ({error}); waiting 60 seconds before retry", file=sys.stderr)
            time.sleep(PAGE_RETRY_DELAY_SECONDS)
            continue
        if is_retryable_http_error(response) or is_browser_error_page(page):
            print("Search page opened a browser/network error screen; waiting 60 seconds before retry", file=sys.stderr)
            time.sleep(PAGE_RETRY_DELAY_SECONDS)
            continue
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if rendered_page_level_ids(page):
                sort_list_by_rating_count(page)
                return
            page.wait_for_timeout(250)
        print("Search page loaded but its level cards are missing; retrying in 3 seconds", file=sys.stderr)
        page.wait_for_timeout(3000)


def rendered_level_links(page) -> list[tuple[str, str, str]]:
    """Read custom demon IDs from rendered demon cards and their proof fields."""
    links = page.locator(".grid-level").evaluate_all("""
      cards => cards.map(card => {
        const fiberKey = Object.keys(card).find(key => key.startsWith('__reactFiber$'));
        let fiber = fiberKey && card[fiberKey];
        while (fiber) {
          const props = fiber.memoizedProps || {};
          if (Number.isInteger(props.ID)) {
            return {id: String(props.ID), rarity: props.rarity, proof: props.proof || ''};
          }
          fiber = fiber.return;
        }
        return null;
      }).filter(Boolean)
    """)
    found: dict[str, tuple[str, str, str]] = {}
    for item in links:
        level_id = item.get("id")
        # GDDL's rendered card prop uses rarity=1 for demons (the card icon
        # itself is rendered as a demon-face logo).
        showcase = str(item.get("proof") or "")
        if level_id and int(level_id) > 3 and item.get("rarity", 0) > 0:
            found.setdefault(level_id, (level_id, f"{BASE_URL}/level/{level_id}", showcase))
    return list(found.values())


def rendered_page_level_ids(page) -> tuple[str, ...]:
    """Read all IDs on the visible list page, including cards we later exclude."""
    ids = page.locator(".grid-level").evaluate_all("""
      cards => cards.map(card => {
        const fiberKey = Object.keys(card).find(key => key.startsWith('__reactFiber$'));
        let fiber = fiberKey && card[fiberKey];
        while (fiber) {
          const id = fiber.memoizedProps?.ID;
          if (Number.isInteger(id)) return String(id);
          fiber = fiber.return;
        }
        return null;
      }).filter(Boolean)
    """)
    return tuple(ids)


def read_rating_spread(page, level_id: str, timeout_ms: int = 8000) -> dict[int, int]:
    """Read the rendered Rating spread histogram, not the paginated submissions."""
    spread = page.locator("section.mt-6.grid.grid-cols-1").first
    deadline = time.monotonic() + timeout_ms / 1000
    scrolled = False
    # Some level pages mount the lower details only when the page is scrolled.
    # Trigger that render even if the panel is not in the DOM yet.
    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    except Exception:
        pass
    while time.monotonic() < deadline:
        if spread.count():
            try:
                if not scrolled:
                    spread.scroll_into_view_if_needed(timeout=1500)
                    scrolled = True
                panel_text = spread.inner_text(timeout=1500).strip()
                if "Rating spread" in panel_text:
                    tables = spread.locator("table")
                    if tables.count():
                        rows = tables.first.evaluate(
                            "table => [...table.rows].map(row => [...row.cells].map(cell => cell.innerText.trim()))"
                        )
                        distribution = {tier: 0 for tier in range(1, 40)}
                        parsed_tiers = 0
                        for cells in rows:
                            if len(cells) < 2:
                                continue
                            tier_match = re.search(r"\d+", cells[0])
                            votes_match = re.search(r"\d[\d,]*", cells[1])
                            if not tier_match or not votes_match:
                                continue
                            tier = int(tier_match.group())
                            votes = int(votes_match.group().replace(",", ""))
                            if 1 <= tier <= 39:
                                distribution[tier] = votes
                                parsed_tiers += 1
                        if parsed_tiers:
                            return distribution
            except Exception:
                # The React panel can be replaced while it hydrates; keep
                # polling the live locator until the short field timeout.
                pass
        page.wait_for_timeout(200)
    raise LevelDataNotRendered(f"Rating spread did not render for level {level_id}")


def parse_page(page, level_id: str, level_url: str, showcase_url: str) -> dict | None:
    try:
        response = page.goto(level_url, wait_until="domcontentloaded", timeout=10000)
    except Exception as error:
        raise PageLoadError(f"navigation failed: {error}") from error
    if is_retryable_http_error(response) or is_browser_error_page(page):
        status = getattr(response, "status", "unknown")
        raise PageLoadError(f"browser/network error screen (HTTP {status})")

    # Confirm this is a rendered level page before treating missing data as a
    # legitimate exclusion (for example, no showcase video).
    level_length = wait_for_level_length(page, level_id)
    if level_length == "Platformer":
        print(f"  skip {level_id}: Platformer levels are excluded", file=sys.stderr)
        return None
    showcases = [showcase_url] if is_youtube_video_url(showcase_url) else []
    rendered_video_links = page.locator("a[href], iframe[src]").evaluate_all(
        "els => els.map(e => e.href || e.src).filter(Boolean)"
    )
    for video_url in rendered_video_links:
        if is_youtube_video_url(video_url) and video_url not in showcases:
            showcases.append(video_url)
    if int(level_id) <= 3 or not showcases:
        return None
    distribution = read_rating_spread(page, level_id)
    total_votes = sum(distribution.values())
    if total_votes < 5:
        print(f"  skip {level_id}: {total_votes} tier votes (<5)", file=sys.stderr)
        return None

    title = page.title().split(" - GD Demon Ladder")[0].strip()
    return {
        "level_id": level_id,
        "name": title,
        "level_length": level_length,
        "level_url": level_url,
        "showcase_links": showcases,
        "difficulty_vote_count": total_votes,
        "difficulty_vote_distribution": dict(sorted(distribution.items())),
        "scrape_complete": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("gdladder_demons.csv"))
    parser.add_argument("--start-url", default=f"{BASE_URL}/search", help="Rendered GDDL search page")
    parser.add_argument("--max-pages", type=int, default=0, help="0 means continue through all list pages")
    parser.add_argument("--max-levels", type=int, default=0, help="0 means process every demon found")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between level pages")
    parser.add_argument("--resume", action="store_true", help="Keep completed rows and retry unfinished levels from page 1")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install browser support with: pip install playwright && playwright install chromium", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["level_id", "name", "level_length", "level_url", "showcase_links", "difficulty_vote_count", "difficulty_vote_distribution", "scrape_complete"]
    saved_rows: dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as existing_file:
            existing_rows = list(csv.DictReader(existing_file))
        # Normalize rows written by earlier runs to integer-key histograms and
        # derive the count only from the histogram, as requested.
        for row in existing_rows:
            normalized_length = normalize_length_label(row.get("level_length", ""))
            if normalized_length == "Platformer":
                continue
            if normalized_length:
                row["level_length"] = normalized_length
            try:
                raw_histogram = row["difficulty_vote_distribution"]
                try:
                    old_histogram = json.loads(raw_histogram)
                except json.JSONDecodeError:
                    old_histogram = ast.literal_eval(raw_histogram)
                histogram = {tier: 0 for tier in range(1, 40)}
                histogram.update({int(tier): int(count) for tier, count in old_histogram.items() if 1 <= int(tier) <= 39})
                row["difficulty_vote_distribution"] = repr(histogram)
                row["difficulty_vote_count"] = str(sum(histogram.values()))
                # Rows written before reading the page's Rating spread lack
                # the displayed length label; re-read them with the current UI.
                row["scrape_complete"] = (
                    row.get("scrape_complete")
                    if row.get("level_length")
                    else "false"
                )
            except (KeyError, TypeError, ValueError, SyntaxError):
                row["difficulty_vote_distribution"] = "{}"
                row["difficulty_vote_count"] = "0"
            if row.get("level_id"):
                saved_rows[row["level_id"]] = {field: row.get(field, "") for field in fields}

    def persist_rows() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_path = args.output.with_name(args.output.name + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(saved_rows.values())
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, args.output)

    # Always leave a valid CSV header on disk before starting network work.
    persist_rows()
    completed_ids = {
        level_id for level_id, row in saved_rows.items()
        if str(row.get("scrape_complete", "true")).lower() == "true"
    }
    rows_written = 0
    attempted = 0
    seen_ids: set[str] = set()
    seen_page_signatures: set[tuple[str, ...]] = set()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        list_page = browser.new_page()
        detail_page = browser.new_page()
        # Sort by rating count descending so likely qualifying levels are
        # encountered early. Qualification still comes from counted tier votes.
        load_rendered_list(list_page, args.start_url)

        page_number = 0
        exhausted_by_limit = False
        empty_page_since = None
        pager_failures = 0
        while True:
            try:
                page_signature = rendered_page_level_ids(list_page)
            except Exception as error:
                page_signature = ()
                print(f"List page could not be read ({error}); waiting to retry", file=sys.stderr)
            if not page_signature:
                empty_page_since = empty_page_since or time.monotonic()
                if time.monotonic() - empty_page_since >= 10:
                    print("List page is missing its cards; retrying without the network-error cooldown", file=sys.stderr)
                    recover_list_page(list_page, args.start_url, max(page_number + 1, 1))
                    empty_page_since = None
                else:
                    list_page.wait_for_timeout(500)
                continue

            empty_page_since = None
            is_new_page = page_signature not in seen_page_signatures
            if is_new_page:
                page_number += 1
                seen_page_signatures.add(page_signature)
                try:
                    candidates = rendered_level_links(list_page)
                except Exception as error:
                    print(f"Could not read demon cards on loaded list page {page_number} ({error}); retrying in 3 seconds", file=sys.stderr)
                    seen_page_signatures.remove(page_signature)
                    page_number -= 1
                    list_page.wait_for_timeout(3000)
                    recover_list_page(list_page, args.start_url, max(page_number + 1, 1))
                    continue
                print(f"List page {page_number}: checking {len(candidates)} custom demon IDs", file=sys.stderr)
                for level_id, url, showcase in candidates:
                    if level_id in seen_ids or level_id in completed_ids:
                        continue
                    seen_ids.add(level_id)
                    if args.max_levels and attempted >= args.max_levels:
                        exhausted_by_limit = True
                        break
                    attempted += 1
                    print(f"[{attempted}] checking {level_id}", file=sys.stderr)
                    row = None
                    navigation_retried = False
                    data_retries = 0
                    while True:
                        try:
                            row = parse_page(detail_page, level_id, url, showcase)
                            break
                        except PageLoadError as error:
                            if not navigation_retried:
                                navigation_retried = True
                                print(f"[{attempted}] {level_id} did not load ({error}); waiting 60 seconds before retry", file=sys.stderr)
                                try:
                                    detail_page.close()
                                except Exception:
                                    pass
                                time.sleep(PAGE_RETRY_DELAY_SECONDS)
                                detail_page = browser.new_page()
                                continue
                            print(f"[{attempted}] {level_id} still did not load after the 60-second retry; skipping", file=sys.stderr)
                            break
                        except LevelDataNotRendered as error:
                            if data_retries < 1:
                                data_retries += 1
                                print(f"[{attempted}] {error}; retrying the loaded level page ({data_retries}/1)", file=sys.stderr)
                                detail_page.wait_for_timeout(1500)
                                continue
                            print(f"[{attempted}] required level data still missing after retry ({error}); skipping", file=sys.stderr)
                            break
                        except Exception as error:
                            if data_retries < 1:
                                data_retries += 1
                                print(f"[{attempted}] page loaded but data read failed ({error}); retrying once", file=sys.stderr)
                                detail_page.wait_for_timeout(1500)
                                continue
                            print(f"[{attempted}] page loaded but required data was missing for {level_id} ({error}); skipping", file=sys.stderr)
                            break
                    try:
                        if row:
                            saved_rows[level_id] = {
                                **row,
                                "showcase_links": json.dumps(row["showcase_links"], ensure_ascii=False),
                                "difficulty_vote_distribution": repr(row["difficulty_vote_distribution"]),
                            }
                            persist_rows()
                            rows_written += 1
                            completed_ids.add(level_id)
                            print(f"[{attempted}] kept {level_id}: {row['difficulty_vote_count']} votes", file=sys.stderr)
                        else:
                            saved_rows.pop(level_id, None)
                            persist_rows()
                            completed_ids.add(level_id)
                            print(f"[{attempted}] skipped {level_id}", file=sys.stderr)
                    except Exception as error:
                        print(f"[{attempted}] could not save {level_id}: {error}; continuing", file=sys.stderr)
                    if args.delay:
                        time.sleep(args.delay)
            else:
                print("List page repeated unexpectedly; retrying its pager", file=sys.stderr)

            if exhausted_by_limit or (args.max_pages and page_number >= args.max_pages):
                break
            pager_failures = 0
            while True:
                try:
                    has_next = click_next_page(list_page)
                    pager_failures = 0
                    break
                except Exception as error:
                    pager_failures += 1
                    print(f"Could not navigate to the next list page ({error}); waiting 60 seconds before retry {pager_failures}", file=sys.stderr)
                    time.sleep(PAGE_RETRY_DELAY_SECONDS)
                    recover_list_page(list_page, args.start_url, max(page_number + 1, 1))
            if not has_next:
                break
        browser.close()
    print(f"Completed {rows_written} demons; CSV checkpoints are in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
