import math
import time
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import random
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString, Tag


DELAY = 3


class DataFetcher:
    BASE_SEARCH_URL = "https://store.steampowered.com/search/results/"
    BASE_APP_URL = "https://store.steampowered.com/app"
    PER_PAGE = 25

    def __init__(
        self,
        db_path: str = "steam_top_100.db",
        target_count: int = 100,
        filter_mode: str = "globaltopsellers",
        request_delay: float = 0.3,
    ):
        self.db_path = db_path
        self.target_count = target_count
        self.filter_mode = filter_mode
        self.request_delay = request_delay

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; SteamDataFetcherForAcademicResearch/1.2; +https://example.com)"
            )
        }

        self._appid_regex = re.compile(r"/(\d+)/")
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS steam_games (
                    appid INTEGER PRIMARY KEY,
                    name TEXT,
                    rank INTEGER,
                    price TEXT,
                    release_date TEXT
                )
                """
            )
            conn.commit()

            cur.execute("PRAGMA table_info(steam_games)")
            existing_cols = {row[1] for row in cur.fetchall()}

            extra_cols = {
                "url": "TEXT",
                "developer": "TEXT",
                "recent_review": "TEXT",
                "overall_review": "TEXT",
                "store_price": "TEXT",
                "about": "TEXT",
                "genres": "TEXT",
                "recent_review_count": "INTEGER",
                "all_review_count": "INTEGER",
                "user_tags": "TEXT",
            }

            for col_name, col_type in extra_cols.items():
                if col_name not in existing_cols:
                    cur.execute(
                        f"ALTER TABLE steam_games ADD COLUMN {col_name} {col_type}"
                    )
                    conn.commit()

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS steam_reviews (
                    review_id TEXT PRIMARY KEY,
                    appid INTEGER,
                    language TEXT,
                    voted_up INTEGER,
                    votes_up INTEGER,
                    votes_funny INTEGER,
                    timestamp_created INTEGER,
                    timestamp_updated INTEGER,
                    playtime_forever INTEGER,
                    playtime_at_review INTEGER,
                    review_text TEXT,
                    author_steamid TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _fetch_search_page(self, page):
        params = {
            "filter": self.filter_mode,
            "page": page,
            "json": 1,
        }

        resp = requests.get(
            self.BASE_SEARCH_URL, params=params, headers=self.headers, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])

    def _extract_appid_from_logo(self, logo_url):
        if not logo_url:
            return None
        m = self._appid_regex.search(logo_url)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def _normalize_search_item(self, raw, rank):
        logo = raw.get("logo", "")
        appid = raw.get("id") or self._extract_appid_from_logo(logo)
        if appid is None:
            return None

        url = raw.get("url") or f"{self.BASE_APP_URL}/{appid}/"

        return {
            "appid": int(appid),
            "name": raw.get("name", ""),
            "rank": rank,
            "price": raw.get("final_formatted") or raw.get("price", ""),
            "release_date": raw.get("release_date", ""),
            "url": url,
        }

    def _upsert_game_from_search(self, game):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO steam_games (appid, name, rank, price, release_date, url)
                VALUES (:appid, :name, :rank, :price, :release_date, :url)
                ON CONFLICT(appid) DO UPDATE SET
                    name = excluded.name,
                    rank = excluded.rank,
                    price = excluded.price,
                    release_date = excluded.release_date,
                    url = excluded.url
                """,
                game,
            )
            conn.commit()
        finally:
            conn.close()

    def _update_game_details(
        self,
        appid,
        developer,
        recent_review,
        overall_review,
        store_price,
        about,
        genres,
        release_date,
        recent_count,
        overall_count,
        user_tags,
    ):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE steam_games
                SET developer           = ?,
                    recent_review       = ?,
                    overall_review      = ?,
                    store_price         = ?,
                    about               = ?,
                    genres              = ?,
                    release_date        = COALESCE(?, release_date),
                    recent_review_count = ?,
                    all_review_count    = ?,
                    user_tags           = ?
                WHERE appid = ?
                """,
                (
                    developer,
                    recent_review,
                    overall_review,
                    store_price,
                    about,
                    genres,
                    release_date,
                    recent_count,
                    overall_count,
                    user_tags,
                    appid,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_and_store(self):
        pages_needed = math.ceil(self.target_count / self.PER_PAGE)
        rank = 1
        stored = 0

        for page in tqdm(range(1, pages_needed + 1), desc="Fetching search pages"):
            self.request_delay = random.random() * DELAY
            try:
                items = self._fetch_search_page(page)
            except Exception as e:
                print(f"Error fetching search page {page}: {e}")
                break

            if not items:
                break

            for item in items:
                if stored >= self.target_count:
                    break

                normalized = self._normalize_search_item(item, rank)
                if normalized is None:
                    continue

                self._upsert_game_from_search(normalized)
                stored += 1
                rank += 1

            if stored >= self.target_count:
                break

            time.sleep(self.request_delay)

    def _build_app_url(self, appid: int) -> str:
        return f"{self.BASE_APP_URL}/{appid}/"

    def _parse_reviews(self, soup: BeautifulSoup):
        recent_review = None
        overall_review = None
        recent_count: Optional[int] = None
        overall_count: Optional[int] = None

        rows = soup.select(".user_reviews_summary_row")
        if not rows:
            return recent_review, overall_review, recent_count, overall_count

        def extract_count(row: Tag):
            span = row.select_one("span.responsive_hidden")
            if not span:
                return None
            text = span.get_text(strip=True)
            digits = re.sub(r"[^\d]", "", text)
            return int(digits) if digits else None

        for row in rows:
            subtitle_el = row.select_one("div.subtitle")
            summary_el = row.select_one("span.game_review_summary")
            if not summary_el:
                continue

            label_text = (subtitle_el.get_text(strip=True).lower()
                          if subtitle_el else "")
            subtitle_classes = subtitle_el.get("class", []) if subtitle_el else []

            summary_text = summary_el.get_text(strip=True)
            count_val = extract_count(row)

            # Recent reviews row
            if "recent" in label_text:
                recent_review = summary_text
                if count_val is not None:
                    recent_count = count_val

            # Overall / all-time row
            elif (
                "all" in subtitle_classes
                or "overall" in label_text
                or "all reviews" in label_text
                or "english reviews" in label_text
            ):
                overall_review = summary_text
                if count_val is not None:
                    overall_count = count_val

        # Fallback: if we still don't have an overall but have rows,treat the last row's summary and count as overall/all-time
        if overall_review is None and rows:
            last_row = rows[-1]
            last_summary = last_row.select_one("span.game_review_summary")
            if last_summary:
                overall_review = last_summary.get_text(strip=True)
            if overall_count is None:
                overall_count = extract_count(last_row)

        return recent_review, overall_review, recent_count, overall_count

    def _parse_developer(self, soup: BeautifulSoup):
        dev_div = soup.select_one("div#developers_list")
        if not dev_div:
            return None

        devs = [a.get_text(strip=True) for a in dev_div.select("a") if a.get_text(strip=True)]
        if not devs:
            return None
        return ", ".join(devs)

    def _parse_store_price(self, soup: BeautifulSoup):
        price_el = soup.select_one(".discount_final_price")
        if not price_el:
            price_el = soup.select_one(".game_purchase_price")
        if not price_el:
            return None
        return price_el.get_text(strip=True)

    def _parse_about(self, soup: BeautifulSoup) -> Optional[str]:
        about_div = soup.select_one("#game_area_description")
        if not about_div:
            return None

        h2 = about_div.find("h2")
        if h2:
            h2.extract()

        text = about_div.get_text(separator="\n", strip=True)
        return text or None

    def _parse_release_date(self, soup: BeautifulSoup) -> Optional[str]:
        date_div = soup.select_one(".release_date .date")
        if not date_div:
            return None
        return date_div.get_text(strip=True)

    def _parse_genres(self, soup: BeautifulSoup) -> Optional[str]:
        # Prefer the specific container if present, fall back to generic details_block
        container = soup.select_one("#genresAndManufacturer")
        if not container:
            blocks = soup.select("div.details_block")
            if not blocks:
                return None
            container = blocks[0]

        genre_label: Optional[Tag] = None
        for b in container.find_all("b"):
            text = b.get_text(strip=True).lower()
            if text.startswith("genre"):
                genre_label = b
                break

        if not genre_label:
            return None

        genres: List[str] = []

        for sib in genre_label.next_siblings:
            if isinstance(sib, Tag) and sib.name == "br":
                break

            if isinstance(sib, Tag):
                if sib.name == "a":
                    txt = sib.get_text(strip=True)
                    if txt:
                        genres.append(txt)
                else:
                    for a in sib.find_all("a"):
                        txt = a.get_text(strip=True)
                        if txt:
                            genres.append(txt)

            elif isinstance(sib, NavigableString):
                parts = [p.strip() for p in str(sib).split(",") if p.strip()]
                genres.extend(parts)

        if not genres:
            return None

        seen = set()
        uniq = []
        for g in genres:
            if g not in seen:
                seen.add(g)
                uniq.append(g)

        return ", ".join(uniq) if uniq else None

    def _parse_user_tags(self, soup: BeautifulSoup) -> Optional[str]:
        container = soup.select_one(".glance_tags.popular_tags")
        if not container:
            return None

        tags: List[str] = []
        # Only <a.app_tag>, skip the add_button div
        for a in container.select("a.app_tag"):
            txt = a.get_text(strip=True)
            if txt:
                tags.append(txt)

        if not tags:
            return None

        return ", ".join(tags)

    def fetch_additional_details_from_store(self, limit=None, update=False):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            if update:
                if limit is not None:
                    cur.execute(
                        "SELECT appid, name, url FROM steam_games ORDER BY rank ASC LIMIT ?",
                        (limit,),
                    )
                else:
                    cur.execute(
                        "SELECT appid, name, url FROM steam_games ORDER BY rank ASC"
                    )
            else:
                if limit is not None:
                    cur.execute(
                        """
                        SELECT appid, name, url
                        FROM steam_games
                        WHERE about IS NULL OR TRIM(about) = ''
                        ORDER BY rank ASC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT appid, name, url
                        FROM steam_games
                        WHERE about IS NULL OR TRIM(about) = ''
                        ORDER BY rank ASC
                        """
                    )

            rows = cur.fetchall()
        finally:
            conn.close()

        for idx, (appid, name, url) in tqdm(
            enumerate(rows, start=1),
            total=len(rows),
            desc="Enriching store details",
        ):
            self.request_delay = random.random() * DELAY
            if not url:
                url = self._build_app_url(appid)

            try:
                url_with_age = url + "?agecheck=1"
                cookies = {
                    "birthtime": "0",
                    "lastagecheckage": "1-January-1970",
                    "mature_content": "1",
                }

                resp = requests.get(
                    url_with_age,
                    headers=self.headers,
                    cookies=cookies,
                    timeout=15,
                    params={"l": "english"},
                )
                resp.raise_for_status()
            except Exception as e:
                print(f"Error fetching app page for {appid}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            try:
                recent_review, overall_review, recent_count, overall_count = self._parse_reviews(soup)
                developer = self._parse_developer(soup)
                store_price = self._parse_store_price(soup)
                about = self._parse_about(soup)
                genres = self._parse_genres(soup)
                release_date = self._parse_release_date(soup)
                user_tags = self._parse_user_tags(soup)

                self._update_game_details(
                    appid=appid,
                    developer=developer,
                    recent_review=recent_review,
                    overall_review=overall_review,
                    store_price=store_price,
                    about=about,
                    genres=genres,
                    release_date=release_date,
                    recent_count=recent_count,
                    overall_count=overall_count,
                    user_tags=user_tags,
                )
            except Exception as e:
                print(f"Error parsing/updating details for {appid}: {e}")
                continue

            time.sleep(self.request_delay)

    def _fetch_reviews_from_api(self, appid, num_reviews, language, filter_type: str = "recent"):
        url = f"https://store.steampowered.com/appreviews/{appid}"
        cursor = "*"
        collected: List[Dict[str, Any]] = []
        per_page = 100

        while len(collected) < num_reviews:
            self.request_delay = random.random() * DELAY
            params = {
                "json": 1,
                "language": language,
                "filter": filter_type,
                "num_per_page": per_page,
                "cursor": cursor,
                "purchase_type": "all",
            }

            try:
                resp = requests.get(url, params=params, headers=self.headers, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                print(f"Error fetching reviews for {appid}: {e}")
                break

            data = resp.json()
            reviews = data.get("reviews", [])
            if not reviews:
                break

            collected.extend(reviews)
            cursor = data.get("cursor")
            if not cursor:
                break

            time.sleep(self.request_delay)

        return collected[:num_reviews]

    def _insert_reviews(self, appid, reviews):
        if not reviews:
            return

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            rows_to_insert = []
            for r in reviews:
                author = r.get("author", {}) or {}
                rows_to_insert.append(
                    (
                        r.get("recommendationid"),
                        appid,
                        r.get("language"),
                        int(bool(r.get("voted_up"))),
                        r.get("votes_up", 0),
                        r.get("votes_funny", 0),
                        r.get("timestamp_created", 0),
                        r.get("timestamp_updated", 0),
                        author.get("playtime_forever", 0),
                        author.get("playtime_at_review", 0),
                        r.get("review", ""),
                        author.get("steamid"),
                    )
                )

            cur.executemany(
                """
                INSERT OR IGNORE INTO steam_reviews (
                    review_id,
                    appid,
                    language,
                    voted_up,
                    votes_up,
                    votes_funny,
                    timestamp_created,
                    timestamp_updated,
                    playtime_forever,
                    playtime_at_review,
                    review_text,
                    author_steamid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.commit()
        finally:
            conn.close()

    def scrape_reviews(
        self,
        reviews_per_game=100,
        language="english",
        filter_type="recent",
        update=False,
    ):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT appid, name FROM steam_games ORDER BY rank ASC")
            games = cur.fetchall()
        finally:
            conn.close()

        for idx, (appid, name) in tqdm(
            enumerate(games, start=1),
            total=len(games),
            desc="Scraping reviews",
        ):
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM steam_reviews WHERE appid = ?",
                    (appid,),
                )
                existing_count = cur.fetchone()[0]
            finally:
                conn.close()

            if not update and existing_count > 0:
                continue

            if update:
                remaining = max(0, reviews_per_game - existing_count)
            else:
                remaining = reviews_per_game

            if remaining <= 0:
                continue

            reviews = self._fetch_reviews_from_api(
                appid=appid,
                num_reviews=remaining,
                language=language,
                filter_type=filter_type,
            )
            self._insert_reviews(appid, reviews)
