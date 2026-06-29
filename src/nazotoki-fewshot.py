import argparse
import base64
import csv
import json
import mimetypes
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from openai import OpenAI

# --- JSON only prompt ---
DEFAULT_PROMPT = (
    "あなたは日本語の『謎解き（ナゾトキ）』を解くアシスタントです。検索は行わず、入力画像と与えられた例だけで解いてください。\n"
    "時間がかかってもいいので、正しい解答をするようにしてください。\n"
    "必ず JSON だけを出力してください（前後に説明文、コードフェンス、箇条書きは禁止）。\n"
    "出力は次のキーを必ず含む単一の JSON オブジェクト：\n"
    '{"answer":"...", "explanation":"...", "confidence":0.0}\n'
    "answer: 最終解答（短い語句）\n"
    "explanation: 画像から読み取れる手掛かりと対応関係を、人が検証できる形で要約\n"
    "confidence: 0.0〜1.0\n"
)

_NUM_RE = re.compile(r"(\d+)")
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/jpeg"


def to_data_url(image_path: Path) -> str:
    mime = guess_mime(image_path)
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def safe_json_load(s: str) -> dict:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty output_text")

    try:
        return json.loads(s)
    except Exception:
        pass

    # code fence除去
    s2 = s.replace("```json", "").replace("```", "").strip()

    # 最初の { 〜 最後の } を抽出
    l = s2.find("{")
    r = s2.rfind("}")
    if l != -1 and r != -1 and l < r:
        return json.loads(s2[l:r + 1])

    raise ValueError("not valid JSON")


def read_first_column_list(path: str) -> list[str]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row[0] if row else "")
    return rows


def load_neighbor1(similar_csv_path: str) -> Dict[int, int]:
    mp: Dict[int, int] = {}
    with open(similar_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or ("query" not in reader.fieldnames) or ("neighbor1" not in reader.fieldnames):
            raise ValueError("similar csv must have columns: query, neighbor1")
        for r in reader:
            if not r.get("query") or not r.get("neighbor1"):
                continue
            mp[int(r["query"])] = int(r["neighbor1"])
    return mp


def extract_id_from_filename(p: Path) -> Optional[int]:
    m = _NUM_RE.search(p.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def iter_images_with_id(root: Path, recursive: bool):
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if p.is_file() and p.suffix.lower() in EXTS:
            qid = extract_id_from_filename(p)
            if qid is not None:
                yield qid, p


def call_with_retry(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    target_image_url: str,
    neighbor_image_url: Optional[str],
    neighbor_answer: str,

    neighbor_explanation: str,
    max_retries: int = 5,
    sleep_sec: float = 0.0,
) -> Tuple[str, str, str, str]:
    delay = 1.0
    last_err = None

    content = [{"type": "input_text", "text": prompt}]

    if neighbor_image_url is not None:
        content.append({"type": "input_text", "text": "【例題（類似問題）】"})
        content.append({"type": "input_image", "image_url": neighbor_image_url})
        content.append({
            "type": "input_text",
            "text": (
                "【例題の解説】\n"
                f"{neighbor_explanation.strip() if neighbor_explanation.strip() else '(解説なし)'}\n\n"
                "【例題の答え】\n"
                f"{neighbor_answer.strip() if neighbor_answer.strip() else '(答えなし)'}\n"
            )
        })

    content.append({"type": "input_text", "text": "【本問】次の画像を解いてください。JSON だけで答えてください。"})
    content.append({"type": "input_image", "image_url": target_image_url})

    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                text={"format": {"type": "json_object"}},
                reasoning={"effort": "high"},
            )

            raw = (resp.output_text or "").strip()
            try:
                obj = safe_json_load(raw)
                answer = str(obj.get("answer", "")).strip()
                explanation = str(obj.get("explanation", "")).strip()
                confidence = str(obj.get("confidence", "")).strip()
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                return answer, explanation, confidence, ""
            except Exception as pe:
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                return raw, "", "", f"ParseError: {pe}"

        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            msg = str(e)

            # ★クォータ切れはリトライしても無意味なので、すぐ返す（途中保存は main 側で行う）
            if "insufficient_quota" in msg:
                return "", "", "", f"InsufficientQuota: {msg}"

            retryable = False
            if status in (429, 500, 502, 503, 504):
                retryable = True
            elif any(k in msg for k in ["429", "Rate limit", "timeout", "temporarily", "Service Unavailable"]):
                retryable = True

            if (not retryable) or (attempt == max_retries):
                break

            time.sleep(delay)
            delay = min(delay * 2, 20)

    return "", "", "", f"{type(last_err).__name__}: {last_err}"


def atomic_write_csv(out_path: Path, fieldnames: List[str], rows: List[dict]) -> Path:
    """
    ★安全保存:
      1) out_path.tmp に書き出し
      2) out_path に置換
    out_path がロックされていても .tmp に結果が残る
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    try:
        tmp_path.replace(out_path)
        return out_path
    except PermissionError:
        print(f"[WARN] 出力先がロックされています: {out_path}", file=sys.stderr)
        print(f"[WARN] 一時ファイルとして保存しました: {tmp_path}", file=sys.stderr)
        return tmp_path


def main():
    ap = argparse.ArgumentParser(
        description="Solve nazotoki with neighbor1 few-shot and output CSV like results1.csv + neighbor1 (autosave)."
    )
    ap.add_argument("--dir", required=True, help="画像ディレクトリ（例: ./images）")
    ap.add_argument("--out", default="results_fewshot.csv", help="出力CSV")
    ap.add_argument("--model", default="gpt-5.1", help="モデル名")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="プロンプト")
    ap.add_argument("--recursive", action="store_true", help="サブディレクトリも探索")
    ap.add_argument("--max-retries", type=int, default=5, help="リトライ回数")
    ap.add_argument("--sleep", type=float, default=0.0, help="各API後に待つ秒数（レート制限回避）")

    # ★追加：何件ごとに保存するか
    ap.add_argument("--save-every", type=int, default=1, help="N件処理ごとに保存（デフォルト: 1）")

    # few-shot用データ
    ap.add_argument("--similar", required=True, help="similar_dinov2_vits14.csv（query,neighbor1）")
    ap.add_argument("--answer", required=True, help="answer.csv（ヘッダ無し1列、id順）")
    ap.add_argument("--kaisetu", required=True, help="kaisetu.csv（ヘッダ無し1列、id順）")

    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] --dir が不正です: {root}", file=sys.stderr)
        sys.exit(1)

    # データ読み込み
    neighbor_map = load_neighbor1(args.similar)
    gold_list = read_first_column_list(args.answer)
    kaisetu_list = read_first_column_list(args.kaisetu)

    gold_by_id = {i + 1: (gold_list[i] or "").strip() for i in range(len(gold_list))}
    kaisetu_by_id = {i + 1: (kaisetu_list[i] or "").strip() for i in range(len(kaisetu_list))}

    # 画像列挙
    items = list(iter_images_with_id(root, args.recursive))
    if not items:
        print(f"[WARN] 画像が見つかりませんでした: {root}")
        sys.exit(0)

    items.sort(key=lambda x: x[0])  # id昇順

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ★結果をメモリに貯めて、save-every ごとに atomic write
    fieldnames = ["id", "filename", "relative_path", "model", "neighbor1", "answer", "explanation", "confidence", "error"]
    rows: List[dict] = []
    processed_since_save = 0

    client = OpenAI()

    for idx, (qid, img_path) in enumerate(items, start=1):
        rel = img_path.relative_to(root)
        nb = neighbor_map.get(qid, None)

        print(f"[{idx}/{len(items)}] id={qid} neighbor1={nb if nb is not None else ''} {rel}")

        # 画像読み込み
        try:
            target_url = to_data_url(img_path)
        except Exception as e:
            rows.append({
                "id": qid,
                "filename": img_path.name,
                "relative_path": str(rel),
                "model": args.model,
                "neighbor1": nb if nb is not None else "",
                "answer": "",
                "explanation": "",
                "confidence": "",
                "error": f"ReadError: {e}",
            })
            processed_since_save += 1
            if processed_since_save >= args.save_every:
                written = atomic_write_csv(out_path, fieldnames, rows)
                print(f"[SAVE] wrote: {written}")
                processed_since_save = 0
            continue

        neighbor_url = None
        neighbor_answer = ""
        neighbor_expl = ""

        if nb is not None:
            found = None
            for ext in [".webp", ".png", ".jpg", ".jpeg", ".gif"]:
                p = root / f"{nb}{ext}"
                if p.exists():
                    found = p
                    break
            if found is None:
                for p in root.glob("*"):
                    if p.is_file() and p.suffix.lower() in EXTS:
                        mid = extract_id_from_filename(p)
                        if mid == nb:
                            found = p
                            break

            if found is not None:
                try:
                    neighbor_url = to_data_url(found)
                except Exception:
                    neighbor_url = None

            neighbor_answer = gold_by_id.get(nb, "")
            neighbor_expl = kaisetu_by_id.get(nb, "")

        ans, expl, conf, err = call_with_retry(
            client,
            model=args.model,
            prompt=args.prompt,
            target_image_url=target_url,
            neighbor_image_url=neighbor_url,
            neighbor_answer=neighbor_answer,
            neighbor_explanation=neighbor_expl,
            max_retries=args.max_retries,
            sleep_sec=args.sleep,
        )

        rows.append({
            "id": qid,
            "filename": img_path.name,
            "relative_path": str(rel),
            "model": args.model,
            "neighbor1": nb if nb is not None else "",
            "answer": ans,
            "explanation": expl,
            "confidence": conf,
            "error": err or "",
        })

        processed_since_save += 1
        if processed_since_save >= args.save_every:
            written = atomic_write_csv(out_path, fieldnames, rows)
            print(f"[SAVE] wrote: {written}")
            processed_since_save = 0

        # ★クォータ切れなら、その時点で保存して終了
        if (err or "").startswith("InsufficientQuota:"):
            written = atomic_write_csv(out_path, fieldnames, rows)
            print(f"[FATAL] quota exhausted. saved: {written}", file=sys.stderr)
            break

    # 最後に必ず保存
    written = atomic_write_csv(out_path, fieldnames, rows)
    print(f"[DONE] CSV 出力: {written}")


if __name__ == "__main__":
    main()
