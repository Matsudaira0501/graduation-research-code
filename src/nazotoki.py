import argparse
import base64
import csv
import json
import mimetypes
import sys
import time
import re
from pathlib import Path

from openai import OpenAI

_NUM_RE = re.compile(r"(\d+)")

DEFAULT_PROMPT = (
    "この画像は日本語の『謎解き（ナゾトキ）』です。検索は行わず、画像の情報だけで解いてください。\n"
    "時間がかかってもいいので、正しい解答をするようにしてください。\n"
    "必ず JSON だけを出力してください（前後に説明文、コードフェンス、箇条書きは付けない）。\n"
    "出力は次のキーを必ず含む単一の JSON オブジェクト：\n"
    '{"answer":"...", "explanation":"...", "confidence":0.0}\n'
    "answer: 最終解答（短い語句）\n"
    "explanation: 根拠の要約（解答の過程を分かりやすく）\n"
    "confidence: 0.0〜1.0\n"
)



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

def iter_images(root: Path, recursive: bool):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                yield p

def _safe_json_load(s: str):
    """
    JSON以外が混ざっても、最初の { から最後の } までを抜いてパースを試す。
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("empty output_text")

    # まず素直に
    try:
        return json.loads(s)
    except Exception:
        pass

    # code fence等を除去
    s2 = s.replace("```json", "").replace("```", "").strip()

    # 最初の { 〜 最後の } を切り出す
    l = s2.find("{")
    r = s2.rfind("}")
    if l != -1 and r != -1 and l < r:
        return json.loads(s2[l:r+1])

    raise ValueError("not valid JSON")

def call_with_retry(client: OpenAI, *, model: str, prompt: str, image_data_url: str, max_retries: int = 5):
    delay = 1.0
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }],
                # ★Responses APIのJSON mode指定
                text={"format": {"type": "json_object"}},
                reasoning={"effort": "high"},
            )

            raw = (resp.output_text or "").strip()
            obj = _safe_json_load(raw)

            answer = str(obj.get("answer", "")).strip()
            explanation = str(obj.get("explanation", "")).strip()
            confidence = obj.get("confidence", "")

            return answer, explanation, confidence, None

        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            msg = str(e)

            is_retryable = False
            if status in (429, 500, 502, 503, 504):
                is_retryable = True
            elif any(code in msg for code in ["429", "Rate limit", "timeout", "temporarily", "Service Unavailable"]):
                is_retryable = True

            if not is_retryable or attempt == max_retries:
                break

            time.sleep(delay)
            delay = min(delay * 2, 20)

    return "", "", "", f"{type(last_err).__name__}: {last_err}"

def sort_key_for_webp_number(path: Path):
    """
    webp のファイル名に含まれる数字を抽出し、数字昇順で処理する。
    webp以外は後ろに回す。
    """
    ext = path.suffix.lower()
    name = path.name.lower()

    if ext == ".webp":
        m = _NUM_RE.search(path.stem)
        num = int(m.group(1)) if m else 10**12
        return (0, num, name)

    return (1, name)


def main():
    parser = argparse.ArgumentParser(description="Solve nazotoki images with OpenAI API and export answers to CSV.")
    parser.add_argument("--dir", type=str, default=".", help="画像があるディレクトリ（デフォルト: カレント）")
    parser.add_argument("--out", type=str, default="answers.csv", help="出力CSVパス（デフォルト: answers.csv）")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="使用モデル（例: gpt-4.1-mini, gpt-4o-mini）")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="モデルに渡す指示文")
    parser.add_argument("--recursive", action="store_true", help="サブディレクトリも再帰的に探索する")
    parser.add_argument("--max-retries", type=int, default=5, help="リトライ回数（デフォルト: 5）")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] --dir が不正です: {root}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    

    images = sorted(iter_images(root, args.recursive), key=sort_key_for_webp_number)
    if not images:
        print(f"[WARN] 画像が見つかりませんでした: {root}")
        sys.exit(0)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "filename", "relative_path", "model", "answer", "explanation", "confidence", "error"],
        )
        writer.writeheader()

        for idx, img_path in enumerate(images, start=1):
            rel = img_path.relative_to(root)
            print(f"[{idx}/{len(images)}] id={idx} {rel}")

            try:
                data_url = to_data_url(img_path)
            except Exception as e:
                writer.writerow({
                    "id": idx,
                    "filename": img_path.name,
                    "relative_path": str(rel),
                    "model": args.model,
                    "answer": "",
                    "explanation": "",
                    "confidence": "",
                    "error": f"ReadError: {e}",
                })
                continue

            answer, explanation, confidence, err = call_with_retry(
                client,
                model=args.model,
                prompt=args.prompt,
                image_data_url=data_url,
                max_retries=args.max_retries,
            )

            writer.writerow({
                "id": idx,
                "filename": img_path.name,
                "relative_path": str(rel),
                "model": args.model,
                "answer": answer,
                "explanation": explanation,
                "confidence": confidence,
                "error": err or "",
            })

    print(f"[DONE] CSV 出力: {out_path}")


if __name__ == "__main__":
    main()
