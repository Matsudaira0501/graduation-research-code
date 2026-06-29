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
EXTS = [".webp", ".png", ".jpg", ".jpeg", ".gif"]


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

    s2 = s.replace("```json", "").replace("```", "").strip()
    l = s2.find("{")
    r = s2.rfind("}")
    if l != -1 and r != -1 and l < r:
        return json.loads(s2[l : r + 1])

    raise ValueError("not valid JSON")


def read_first_column_list(path: str) -> List[str]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row[0] if row else "")
    return rows


def read_dict_csv_auto_encoding(path: str) -> List[dict]:
    """
    Excel/Windows由来のCSV（cp932等）を考慮して辞書で読む。
    """
    encodings_to_try = ["utf-8-sig", "cp932", "shift_jis"]
    last_err = None
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("failed to read csv")


def parse_neighbors_field(s: str) -> List[Tuple[int, str]]:
    """
    例: "38,0.188;51,0.188;55,0.188;60,0.188"
        "9,0.111"
    戻り値: [(38,"0.188"), (51,"0.188"), ...]
    """
    s = (s or "").strip()
    if not s:
        return []
    out: List[Tuple[int, str]] = []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        # "id,score" を想定（scoreが無い場合も許容）
        toks = [t.strip() for t in p.split(",")]
        if not toks or not toks[0].isdigit():
            continue
        nid = int(toks[0])
        score = toks[1] if len(toks) >= 2 else ""
        out.append((nid, score))
    # 重複除去（順序維持）
    seen = set()
    uniq: List[Tuple[int, str]] = []
    for nid, sc in out:
        if nid in seen:
            continue
        seen.add(nid)
        uniq.append((nid, sc))
    return uniq


def load_neighbors_from_top1_ties_csv(path: str) -> Dict[int, List[Tuple[int, str]]]:
    """
    jaccard_top1_ties_pke.csv 等の
      - query
      - top1_neighbors  (= C列)
    を読む。
    """
    rows = read_dict_csv_auto_encoding(path)
    if not rows:
        raise ValueError("neighbors csv is empty")

    # 必須カラム名（あなたのCSVに合わせている）
    sample_keys = set(rows[0].keys())
    if "query" not in sample_keys:
        raise ValueError("neighbors csv must have column: query")
    if "top1_neighbors" not in sample_keys:
        raise ValueError("neighbors csv must have column: top1_neighbors (C列)")

    mp: Dict[int, List[Tuple[int, str]]] = {}
    for r in rows:
        q = (r.get("query") or "").strip()
        if not q.isdigit():
            continue
        qid = int(q)
        mp[qid] = parse_neighbors_field(r.get("top1_neighbors", ""))
    return mp


def find_image_by_id(root: Path, qid: int) -> Optional[Path]:
    """
    基本: {id}.webp を想定（ただし拡張子はフォールバック）
    """
    for ext in EXTS:
        p = root / f"{qid}{ext}"
        if p.exists():
            return p

    # fallback: stem内数字一致（直下のみ）
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        m = _NUM_RE.search(p.stem)
        if m and int(m.group(1)) == qid:
            return p

    return None


def load_existing_results_pairs(path: str) -> Dict[Tuple[int, int], dict]:
    """
    出力CSV（複数類題対応版）を読み、(id, neighbor_id) -> row を返す。
    """
    mp: Dict[Tuple[int, int], dict] = {}
    p = Path(path)
    if not p.exists():
        return mp
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mp
        for r in reader:
            rid = (r.get("id") or "").strip()
            nid = (r.get("neighbor_id") or "").strip()
            if not rid.isdigit() or not nid.isdigit():
                continue
            mp[(int(rid), int(nid))] = r
    return mp


def should_process_row(row: Optional[dict]) -> bool:
    """
    再実行する条件：
    - 行が無い
    - answer が空
    - error が入っている
    """
    if row is None:
        return True
    ans = (row.get("answer") or "").strip()
    err = (row.get("error") or "").strip()
    if ans == "":
        return True
    if err != "":
        return True
    return False


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
) -> Tuple[str, str, str, str, bool]:
    """
    Returns: (answer, explanation, confidence, error, fatal_stop)
    fatal_stop=True の場合、insufficient_quota 等で停止推奨。
    """
    delay = 1.0
    last_err = None

    content = [{"type": "input_text", "text": prompt}]

    if neighbor_image_url is not None:
        content.append({"type": "input_text", "text": "【例題（類似問題）】"})
        content.append({"type": "input_image", "image_url": neighbor_image_url})
        content.append(
            {
                "type": "input_text",
                "text": (
                    "【例題の解説】\n"
                    f"{neighbor_explanation.strip() if neighbor_explanation.strip() else '(解説なし)'}\n\n"
                    "【例題の答え】\n"
                    f"{neighbor_answer.strip() if neighbor_answer.strip() else '(答えなし)'}\n"
                ),
            }
        )
    else:
        # 類題画像が無い場合でも、最低限テキストは渡す（答え/解説があるなら）
        if neighbor_answer.strip() or neighbor_explanation.strip():
            content.append({"type": "input_text", "text": "【例題（類似問題）※画像なし】"})
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "【例題の解説】\n"
                        f"{neighbor_explanation.strip() if neighbor_explanation.strip() else '(解説なし)'}\n\n"
                        "【例題の答え】\n"
                        f"{neighbor_answer.strip() if neighbor_answer.strip() else '(答えなし)'}\n"
                    ),
                }
            )

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
                return answer, explanation, confidence, "", False
            except Exception as pe:
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                return raw, "", "", f"ParseError: {pe}", False

        except Exception as e:
            last_err = e
            msg = str(e)

            # クォータ切れは即停止
            if "insufficient_quota" in msg:
                return "", "", "", f"InsufficientQuota: {msg}", True

            status = getattr(e, "status_code", None)
            retryable = False
            if status in (429, 500, 502, 503, 504):
                retryable = True
            elif any(k in msg for k in ["429", "Rate limit", "timeout", "temporarily", "Service Unavailable"]):
                retryable = True

            if (not retryable) or (attempt == max_retries):
                break

            time.sleep(delay)
            delay = min(delay * 2, 20)

    return "", "", "", f"{type(last_err).__name__}: {last_err}", False


def atomic_write_csv(out_path: Path, fieldnames: List[str], rows_in_order: List[dict]) -> Path:
    """
    途中経過を確実に残すための安全な書き込み：
      1) out_path.tmp に書く
      2) replace で out_path に置換
    out_path がロックされていて replace に失敗しても .tmp が残る。
    戻り値: 実際に書き込まれたパス（out_path か tmp_path）
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_in_order:
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
        description=(
            "Use ALL similar problems listed in neighbors-csv (C列=top1_neighbors). "
            "If multiple neighbors exist, solve the target once per neighbor (one-by-one ICL)."
        )
    )
    ap.add_argument("--dir", required=True, help="画像ディレクトリ（例: ./images）")

    # ★これが「C列に類題が入っているCSV」：jaccard_top1_ties_pke.csv 等
    ap.add_argument("--neighbors-csv", required=True, help="類題CSV（query, top1_neighbors=C列）")

    ap.add_argument("--out", required=True, help="出力CSV（新規/追記リジューム）")
    ap.add_argument("--start-id", type=int, default=1, help="このIDから実行（デフォルト: 1）")
    ap.add_argument("--end-id", type=int, default=10**9, help="このIDまで（デフォルト: 無制限）")

    ap.add_argument("--model", default="gpt-5.1", help="モデル名")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="プロンプト")
    ap.add_argument("--max-retries", type=int, default=5, help="リトライ回数")
    ap.add_argument("--sleep", type=float, default=0.0, help="各API後に待つ秒数")

    ap.add_argument("--answer", required=True, help="answer.csv（ヘッダ無し1列、id順）")
    ap.add_argument("--kaisetu", required=True, help="kaisetu.csv（ヘッダ無し1列、id順）")

    ap.add_argument("--save-every", type=int, default=1, help="N件（=1ペア）処理ごとにCSV保存（デフォルト: 1）")

    args = ap.parse_args()

    client = OpenAI()
    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] --dir が不正です: {root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out).resolve()

    neighbor_map = load_neighbors_from_top1_ties_csv(args.neighbors_csv)
    if not neighbor_map:
        print("[ERROR] neighbors-csv has no usable rows", file=sys.stderr)
        sys.exit(1)

    gold_list = read_first_column_list(args.answer)
    kaisetu_list = read_first_column_list(args.kaisetu)
    gold_by_id = {i + 1: (gold_list[i] or "").strip() for i in range(len(gold_list))}
    kaisetu_by_id = {i + 1: (kaisetu_list[i] or "").strip() for i in range(len(kaisetu_list))}

    # (id, neighbor_id) 単位でリジューム
    existing = load_existing_results_pairs(str(out_path))

    fieldnames = [
        "id",
        "filename",
        "relative_path",
        "model",
        "neighbor_id",
        "neighbor_score",
        "neighbor_filename",
        "neighbor_relative_path",
        "answer",
        "explanation",
        "confidence",
        "error",
    ]

    updated: Dict[Tuple[int, int], dict] = {k: dict(v) for k, v in existing.items()}
    processed_since_save = 0

    def save_now():
        rows = []
        for (qid, nid) in sorted(updated.keys()):
            rows.append({k: updated[(qid, nid)].get(k, "") for k in fieldnames})
        written = atomic_write_csv(out_path, fieldnames, rows)
        print(f"[SAVE] wrote: {written}")

    # 対象 query を neighbors-csv から決める
    qids = sorted(neighbor_map.keys())

    for qid in qids:
        if qid < args.start_id or qid > args.end_id:
            continue

        neighbors = neighbor_map.get(qid, [])
        if not neighbors:
            # C列が空の行は何もしない（要件: 「C列にある類題すべてで」）
            continue

        img_path = find_image_by_id(root, qid)
        if img_path is None:
            # 全neighbor分に対して「画像なしエラー行」を残す（再現性のため）
            for nid, sc in neighbors:
                key = (qid, nid)
                row = updated.get(key, {})
                row.update(
                    {
                        "id": str(qid),
                        "filename": "",
                        "relative_path": "",
                        "model": args.model,
                        "neighbor_id": str(nid),
                        "neighbor_score": sc,
                        "neighbor_filename": "",
                        "neighbor_relative_path": "",
                        "answer": "",
                        "explanation": "",
                        "confidence": "",
                        "error": f"ImageNotFound: id={qid}",
                    }
                )
                updated[key] = row
                processed_since_save += 1
                if processed_since_save >= args.save_every:
                    save_now()
                    processed_since_save = 0
            continue

        rel = img_path.relative_to(root)

        try:
            target_url = to_data_url(img_path)
        except Exception as e:
            for nid, sc in neighbors:
                key = (qid, nid)
                row = updated.get(key, {})
                row.update(
                    {
                        "id": str(qid),
                        "filename": img_path.name,
                        "relative_path": str(rel),
                        "model": args.model,
                        "neighbor_id": str(nid),
                        "neighbor_score": sc,
                        "neighbor_filename": "",
                        "neighbor_relative_path": "",
                        "answer": "",
                        "explanation": "",
                        "confidence": "",
                        "error": f"ReadError: {e}",
                    }
                )
                updated[key] = row
                processed_since_save += 1
                if processed_since_save >= args.save_every:
                    save_now()
                    processed_since_save = 0
            continue

        # ★複数類題がある場合：類題ごとに1回ずつ推論して出力行も別にする
        for nid, sc in neighbors:
            key = (qid, nid)
            row = updated.get(key)

            if not should_process_row(row):
                continue

            neighbor_url = None
            nb_path = find_image_by_id(root, nid)
            nb_rel = ""
            nb_name = ""
            if nb_path is not None:
                nb_name = nb_path.name
                try:
                    nb_rel = str(nb_path.relative_to(root))
                except Exception:
                    nb_rel = ""
                try:
                    neighbor_url = to_data_url(nb_path)
                except Exception:
                    neighbor_url = None

            neighbor_answer = gold_by_id.get(nid, "")
            neighbor_expl = kaisetu_by_id.get(nid, "")

            print(f"[RUN] id={qid} neighbor={nid} score={sc} file={img_path.name}")

            ans, expl, conf, err, fatal = call_with_retry(
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

            if row is None:
                row = {}

            row.update(
                {
                    "id": str(qid),
                    "filename": img_path.name,
                    "relative_path": str(rel),
                    "model": args.model,
                    "neighbor_id": str(nid),
                    "neighbor_score": sc,
                    "neighbor_filename": nb_name,
                    "neighbor_relative_path": nb_rel,
                    "answer": ans,
                    "explanation": expl,
                    "confidence": conf,
                    "error": err or "",
                }
            )
            updated[key] = row

            processed_since_save += 1
            if processed_since_save >= args.save_every:
                save_now()
                processed_since_save = 0

            if fatal:
                print("[FATAL] API quota exhausted (insufficient_quota). Saving and stopping.", file=sys.stderr)
                save_now()
                return

    save_now()
    print("[DONE] finished.")


if __name__ == "__main__":
    main()
