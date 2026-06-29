import argparse
import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


CATEGORY_URL = "https://nazotoki-concierge.com/nazotoki-mondaishu/archives/category/level2"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class Post:
    url: str
    title: str
    published: date
    image_url: str


def parse_jp_date(text: str) -> Optional[date]:
    """
    '2025年11月6日' のような日本語日付を date に変換
    """
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return date(y, mo, d)


def slug_from_url(u: str) -> str:
    # .../archives/2605 みたいなURLから 2605 を取る。取れなければ末尾パスを使う
    path = urlparse(u).path.rstrip("/")
    last = path.split("/")[-1]
    return re.sub(r"[^0-9A-Za-z_-]+", "_", last) or "post"


def get_soup(session: requests.Session, url: str, *, referer: Optional[str], timeout: int) -> BeautifulSoup:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)",
    }
    if referer:
        headers["Referer"] = referer
    r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def collect_post_urls_from_category_page(soup: BeautifulSoup) -> List[str]:
    """
    カテゴリ一覧ページから、記事URL（.../archives/<数字>）を抽出
    """
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # 例: https://.../archives/2605
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


def infer_last_page_num(soup: BeautifulSoup) -> Optional[int]:
    """
    ページングの最後のページ番号を推定（見つからなければ None）
    """
    # /page/20 のようなリンクがあるはずなので最大を拾う
    max_n = None
    for a in soup.find_all("a", href=True):
        m = re.search(r"/page/(\d+)/?$", a["href"])
        if m:
            n = int(m.group(1))
            max_n = n if max_n is None else max(max_n, n)
    return max_n


def extract_post_info(session: requests.Session, post_url: str, *, timeout: int) -> Optional[Tuple[str, date, str]]:
    """
    記事ページから (title, published_date, image_url) を抽出。
    画像は article 内の /wp-content/uploads/ 配下リンク/画像のうち最初のものを採用。
    """
    soup = get_soup(session, post_url, referer=CATEGORY_URL, timeout=timeout)

    # タイトル: h1/h2 のどれかにある想定（WordPress）
    title = None
    for tag in ["h1", "h2"]:
        h = soup.find(tag)
        if h and h.get_text(strip=True):
            title = h.get_text(strip=True)
            break
    title = title or slug_from_url(post_url)

    # 日付: 日本語日付が本文に出る（例: 2025年11月6日）
    published = parse_jp_date(soup.get_text(" ", strip=True))
    if not published:
        return None

    # 記事本文っぽい範囲を優先（article があればそこ）
    scope = soup.find("article") or soup

    # 画像URL候補を抽出（href と src の両方）
    candidates = []

    for a in scope.find_all("a", href=True):
        href = a["href"]
        if "/wp-content/uploads/" in href and href.lower().endswith(IMG_EXTS):
            candidates.append(href)

    for img in scope.find_all("img", src=True):
        src = img["src"]
        if "/wp-content/uploads/" in src and src.lower().endswith(IMG_EXTS):
            candidates.append(src)

    # 重複除去
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    if not candidates:
        return None

    image_url = candidates[0]
    return title, published, image_url


def download_file(session: requests.Session, url: str, out_path: Path, *, referer: str, timeout: int) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)",
        "Referer": referer,
    }
    with session.get(url, headers=headers, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="level2_images", help="保存先フォルダ")
    ap.add_argument("--sleep", type=float, default=0.8, help="アクセス間隔（秒）")
    ap.add_argument("--timeout", type=int, default=20, help="HTTPタイムアウト（秒）")
    ap.add_argument("--start-index", type=int, default=1, help="番号の開始値（例: 1 なら 001 から）")
    ap.add_argument("--max-pages", type=int, default=0, help="カテゴリページを最大何ページまで見るか（0=自動推定）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # まず1ページ目
    soup1 = get_soup(session, CATEGORY_URL, referer=None, timeout=args.timeout)
    last_page = infer_last_page_num(soup1)
    if args.max_pages and args.max_pages > 0:
        total_pages = args.max_pages
    else:
        total_pages = last_page or 1

    # カテゴリ全ページから記事URL収集
    post_urls = []
    for page in range(1, total_pages + 1):
        page_url = CATEGORY_URL if page == 1 else f"{CATEGORY_URL}/page/{page}"
        try:
            soup = get_soup(session, page_url, referer=CATEGORY_URL, timeout=args.timeout)
        except Exception as e:
            print(f"[WARN] ページ取得失敗: {page_url} ({e})")
            continue
        urls = collect_post_urls_from_category_page(soup)
        post_urls.extend(urls)
        time.sleep(args.sleep)

    # 重複除去
    seen = set()
    post_urls = [u for u in post_urls if not (u in seen or seen.add(u))]

    # 各記事から日付・画像URLを抽出
    posts: List[Post] = []
    for u in tqdm(post_urls, desc="Collect posts"):
        try:
            info = extract_post_info(session, u, timeout=args.timeout)
            if info is None:
                continue
            title, published, image_url = info
            posts.append(Post(url=u, title=title, published=published, image_url=image_url))
        except Exception:
            # 個別失敗はスキップして継続
            continue
        time.sleep(args.sleep)

    if not posts:
        print("[ERROR] 記事が取得できませんでした。サイト構造が変わった可能性があります。")
        return

    # 古い順にソート（同日ならURLで安定化）
    posts.sort(key=lambda p: (p.published, p.url))

    # 連番で保存 + マニフェストCSV
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "date", "title", "post_url", "image_url", "filename"])

        idx = args.start_index
        for p in tqdm(posts, desc="Download images"):
            ext = os.path.splitext(urlparse(p.image_url).path)[1].lower()
            no = f"{idx:03d}"
            filename = f"{no}{ext}"
            out_path = out_dir / filename

            if out_path.exists() and out_path.stat().st_size > 0:
                w.writerow([no, p.published.isoformat(), p.title, p.url, p.image_url, filename])
                idx += 1
                continue

            try:
                download_file(session, p.image_url, out_path, referer=p.url, timeout=args.timeout)
                w.writerow([no, p.published.isoformat(), p.title, p.url, p.image_url, filename])
                idx += 1
            except Exception as e:
                print(f"[WARN] ダウンロード失敗: {p.image_url} ({e})")
                # 失敗しても番号は詰めたい場合は idx を増やさない
                # ここでは「欠番を作らない」方針で増やしません
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass

            time.sleep(args.sleep)

    print(f"[DONE] 保存先: {out_dir.resolve()}")
    print(f"[DONE] 一覧CSV: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
