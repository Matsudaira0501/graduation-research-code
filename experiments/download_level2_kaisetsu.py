import argparse
import csv
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CATEGORY_URL = "https://nazotoki-concierge.com/nazotoki-mondaishu/archives/category/level2"


@dataclass(frozen=True)
class Item:
    no: int
    title: str
    published: date
    puzzle_url: str
    kaisetsu_url: str
    kaisetsu_text: str


def parse_jp_date(text: str) -> Optional[date]:
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return date(y, mo, d)


def get_soup(session: requests.Session, url: str, *, referer: Optional[str], timeout: int) -> BeautifulSoup:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; nazotoki-downloader/1.0)"}
    if referer:
        headers["Referer"] = referer
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def infer_total_pages(category_soup: BeautifulSoup) -> int:
    # 例: /page/20 のようなリンクがあるので最大を拾う
    max_n = 1
    for a in category_soup.find_all("a", href=True):
        m = re.search(r"/page/(\d+)/?$", a["href"])
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n


def collect_puzzle_urls_from_category_page(soup: BeautifulSoup) -> List[str]:
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # カテゴリページ内の「LEVEL2 タイトル」リンクは /archives/<数字>
        if re.search(r"/nazotoki-mondaishu/archives/\d+/?$", href):
            urls.append(href)
    # 重複除去（順序維持）
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_title_and_date_from_puzzle(puzzle_soup: BeautifulSoup) -> Tuple[str, date]:
    # タイトルは h2/h1 のどちらかに入っていることが多い
    title = None
    for tag in ["h2", "h1"]:
        h = puzzle_soup.find(tag)
        if h and h.get_text(strip=True):
            title = h.get_text(strip=True)
            break
    title = title or "unknown"

    published = parse_jp_date(puzzle_soup.get_text(" ", strip=True))
    if not published:
        raise ValueError("公開日を抽出できませんでした。")

    return title, published


def find_kaisetsu_link(puzzle_soup: BeautifulSoup, puzzle_url: str) -> Optional[str]:
    # 「解説を見る」のリンクは puzzle ページ内に存在する :contentReference[oaicite:3]{index=3}
    candidates = []
    for a in puzzle_soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if "解説を見る" in text and "/nazotoki-mondaishu/archives/kaisetsu/" in href:
            candidates.append(urljoin(puzzle_url, href))
    return candidates[0] if candidates else None


def extract_kaisetsu_text(kaisetsu_soup: BeautifulSoup, title: str) -> str:
    """
    解説ページは「タイトル」「日付」の下に解説本文が続き、
    その後に「レベル別謎解き一覧」が来る形式なので、
    全文テキストから [日付の直後]〜[レベル別謎解き一覧の直前] を抜き出す。 :contentReference[oaicite:4]{index=4}
    """
    full = kaisetsu_soup.get_text("\n", strip=True)

    # タイトル位置を起点にする（タイトルが複数回出る場合もあるので最初でOK）
    tpos = full.find(title)
    if tpos == -1:
        tpos = 0

    # タイトル以降で日付を探す
    sub = full[tpos:]
    m = re.search(r"\d{4}年\d{1,2}月\d{1,2}日", sub)
    if not m:
        raise ValueError("解説ページで日付を見つけられませんでした。")
    date_str = m.group(0)
    start = tpos + m.end()

    # 終端（「レベル別謎解き一覧」）
    end = full.find("レベル別謎解き一覧", start)
    if end == -1:
        # フォールバック（ページ末まで）
        end = len(full)

    body = full[start:end].strip()

    # 先頭に「解説」等が混ざることがあるので軽く整形
    lines = [ln.strip() for ln in body.splitlines()]
    cleaned = []
    for ln in lines:
        if not ln:
            continue
        if ln in ("解説",):
            continue
        # ナビゲーション語が紛れた場合の保険（必要なら追加）
        if ln.startswith("LEVEL") and len(ln) <= 10:
            continue
        cleaned.append(ln)

    # 段落感を残す：元が空行区切りのことが多いので、連続文は \n\n で結合
    return "\n\n".join(cleaned).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="level2_kaisetsu.csv", help="出力CSVパス")
    ap.add_argument("--sleep", type=float, default=1.0, help="アクセス間隔（秒）")
    ap.add_argument("--timeout", type=int, default=20, help="HTTPタイムアウト（秒）")
    ap.add_argument("--max-pages", type=int, default=0, help="カテゴリページを最大何ページ見るか（0=自動）")
    ap.add_argument("--start-index", type=int, default=1, help="連番開始（例: 1 -> 001）")
    args = ap.parse_args()

    session = requests.Session()

    # カテゴリ1ページ目から総ページ数推定（例: 20） :contentReference[oaicite:5]{index=5}
    soup1 = get_soup(session, CATEGORY_URL, referer=None, timeout=args.timeout)
    total_pages = infer_total_pages(soup1)
    if args.max_pages and args.max_pages > 0:
        total_pages = min(total_pages, args.max_pages)

    # 問題URL収集
    puzzle_urls: List[str] = []
    for p in range(1, total_pages + 1):
        page_url = CATEGORY_URL if p == 1 else f"{CATEGORY_URL}/page/{p}"
        try:
            soup = get_soup(session, page_url, referer=CATEGORY_URL, timeout=args.timeout)
            puzzle_urls.extend(collect_puzzle_urls_from_category_page(soup))
        except Exception as e:
            print(f"[WARN] カテゴリページ取得失敗: {page_url} ({e})")
        time.sleep(args.sleep)

    # 重複除去
    seen = set()
    puzzle_urls = [u for u in puzzle_urls if not (u in seen or seen.add(u))]

    rows = []
    for puzzle_url in tqdm(puzzle_urls, desc="Fetch puzzles"):
        try:
            puzzle_soup = get_soup(session, puzzle_url, referer=CATEGORY_URL, timeout=args.timeout)
            title, published = extract_title_and_date_from_puzzle(puzzle_soup)

            kaisetsu_url = find_kaisetsu_link(puzzle_soup, puzzle_url)
            if not kaisetsu_url:
                print(f"[WARN] 解説リンクが見つかりません: {puzzle_url}")
                continue

            kaisetsu_soup = get_soup(session, kaisetsu_url, referer=puzzle_url, timeout=args.timeout)
            kaisetsu_text = extract_kaisetsu_text(kaisetsu_soup, title)

            rows.append((published, title, puzzle_url, kaisetsu_url, kaisetsu_text))

        except Exception as e:
            print(f"[WARN] 失敗: {puzzle_url} ({e})")

        time.sleep(args.sleep)

    if not rows:
        print("[ERROR] 取得できたデータがありませんでした。")
        return

    # 古い順（昇順）にソート
    rows.sort(key=lambda x: (x[0], x[2]))

    # CSV出力（Excel で開きやすい utf-8-sig）
    out_path = args.out
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "date", "title", "puzzle_url", "kaisetsu_url", "kaisetsu_text"])
        idx = args.start_index
        for published, title, puzzle_url, kaisetsu_url, kaisetsu_text in rows:
            w.writerow([f"{idx:03d}", published.isoformat(), title, puzzle_url, kaisetsu_url, kaisetsu_text])
            idx += 1

    print(f"[DONE] 出力: {out_path}")


if __name__ == "__main__":
    main()
