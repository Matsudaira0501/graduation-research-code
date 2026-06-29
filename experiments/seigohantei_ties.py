# seigohantei_allrows.py
import argparse
import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional, List


# ========= Normalization / matching =========

def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:  # ァ..ヶ
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def remove_punct_and_symbols(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in ("P", "S"))


def normalize_basic(s: str, *, ignore_symbols: bool = False) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    if ignore_symbols:
        s = remove_punct_and_symbols(s)
    return s


@lru_cache(maxsize=4096)
def to_reading_hiragana(s: str) -> str:
    try:
        from pykakasi import kakasi
    except Exception as e:
        raise RuntimeError(
            "漢字→読み一致を使うには pykakasi が必要です。\n"
            "次を実行してください: pip install pykakasi"
        ) from e

    kks = kakasi()
    kks.setMode("J", "H")
    kks.setMode("K", "H")
    kks.setMode("H", "H")
    conv = kks.getConverter()
    return conv.do(s)


def normalize_for_match(s: str, *, mode: str, ignore_symbols: bool = False) -> str:
    x = normalize_basic(s, ignore_symbols=ignore_symbols)

    if mode == "exact":
        return x

    x = kata_to_hira(x)
    if mode == "kana":
        return x

    if mode == "reading":
        return to_reading_hiragana(x)

    raise ValueError(f"Unknown mode: {mode}")


def judge(gold: str, pred: str, *, match: str, ignore_symbols: bool) -> tuple[bool, str, str, str, str]:
    g_exact = normalize_for_match(gold, mode="exact", ignore_symbols=ignore_symbols)
    p_exact = normalize_for_match(pred,  mode="exact", ignore_symbols=ignore_symbols)

    if match == "exact":
        g_cmp, p_cmp = g_exact, p_exact
        return (g_cmp == p_cmp), g_exact, p_exact, g_cmp, p_cmp

    if match == "kana":
        g_cmp = normalize_for_match(gold, mode="kana", ignore_symbols=ignore_symbols)
        p_cmp = normalize_for_match(pred,  mode="kana", ignore_symbols=ignore_symbols)
        return (g_cmp == p_cmp), g_exact, p_exact, g_cmp, p_cmp

    if match == "reading":
        g_cmp = normalize_for_match(gold, mode="reading", ignore_symbols=ignore_symbols)
        p_cmp = normalize_for_match(pred,  mode="reading", ignore_symbols=ignore_symbols)
        return (g_cmp == p_cmp), g_exact, p_exact, g_cmp, p_cmp

    if match == "reading_or_exact":
        g_cmp = normalize_for_match(gold, mode="reading", ignore_symbols=ignore_symbols)
        p_cmp = normalize_for_match(pred,  mode="reading", ignore_symbols=ignore_symbols)
        return ((g_exact == p_exact) or (g_cmp == p_cmp)), g_exact, p_exact, g_cmp, p_cmp

    raise ValueError(match)


# ========= CSV reading =========

def read_csv_column(path: str, col_name: Optional[str] = None) -> List[str]:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_err = None

    for enc in encodings:
        try:
            rows: List[str] = []
            with open(path, "r", encoding=enc, newline="") as f:
                if col_name is None:
                    reader = csv.reader(f)
                    for row in reader:
                        rows.append(row[0] if row else "")
                    return rows

                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV header not found (DictReader needs header row).")

                if col_name not in reader.fieldnames:
                    raise ValueError(f"Column '{col_name}' not found. Available={reader.fieldnames}")

                for r in reader:
                    rows.append(r.get(col_name, "") or "")
                return rows
        except UnicodeDecodeError as e:
            last_err = e

    raise last_err or RuntimeError("Failed to read CSV with known encodings.")


def open_dictreader_auto(path: str):
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_err = None
    for enc in encodings:
        try:
            f = open(path, "r", encoding=enc, newline="")
            r = csv.DictReader(f)
            if r.fieldnames is None:
                f.close()
                raise ValueError("pred CSV must have a header row.")
            return f, r
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err or RuntimeError("Failed to read pred CSV with known encodings.")


# ========= Main =========

def main():
    ap = argparse.ArgumentParser(
        description="Judge correctness for ALL prediction rows (no accuracy)."
    )
    ap.add_argument("--gold", required=True, help="正解CSV（1列 / または --gold_col 指定でヘッダ付き）")
    ap.add_argument("--pred", required=True, help="予測CSV（ヘッダ付き・複数行/ID）")
    ap.add_argument("--out", default="pred_with_judgement.csv", help="出力CSV（全行に正誤列を追加）")

    ap.add_argument("--gold_col", default=None, help="gold がヘッダ付きの場合の列名（未指定なら1列CSV）")
    ap.add_argument("--pred_id_col", default="id", help="pred のID列名（デフォルト: id）")
    ap.add_argument("--pred_answer_col", default="answer", help="pred の回答列名（デフォルト: answer）")

    ap.add_argument(
        "--match",
        choices=["exact", "kana", "reading", "reading_or_exact"],
        default="reading_or_exact",
        help="一致判定方式（デフォルト: reading_or_exact）",
    )
    ap.add_argument("--ignore_symbols", action="store_true", help="句読点/括弧/記号を無視")

    args = ap.parse_args()

    gold_list = read_csv_column(args.gold, args.gold_col)
    gold_by_id = {i + 1: gold_list[i] for i in range(len(gold_list))}

    pred_path = Path(args.pred).resolve()
    out_path = Path(args.out).resolve()

    pf, reader = open_dictreader_auto(str(pred_path))
    fieldnames_in = reader.fieldnames[:]  # type: ignore

    # 追加列
    add_cols = [
        "gold",
        "gold_exact",
        "pred_exact",
        "gold_cmp",
        "pred_cmp",
        "is_correct",
        "judge_error",
    ]
    fieldnames_out = fieldnames_in + [c for c in add_cols if c not in fieldnames_in]

    with pf, open(out_path, "w", encoding="utf-8-sig", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames_out)
        w.writeheader()

        for row in reader:
            rid = (row.get(args.pred_id_col) or "").strip()
            pred_ans = (row.get(args.pred_answer_col) or "").strip()

            judge_error = ""
            if not rid.isdigit():
                # id不正でも行は残す
                row["gold"] = ""
                row["gold_exact"] = ""
                row["pred_exact"] = normalize_for_match(pred_ans, mode="exact", ignore_symbols=args.ignore_symbols)
                row["gold_cmp"] = ""
                row["pred_cmp"] = ""
                row["is_correct"] = "False"
                row["judge_error"] = f"BadID: {rid}"
                w.writerow(row)
                continue

            i = int(rid)
            gold = gold_by_id.get(i, "")
            try:
                ok, g_exact, p_exact, g_cmp, p_cmp = judge(
                    gold, pred_ans, match=args.match, ignore_symbols=args.ignore_symbols
                )
                row["gold"] = gold
                row["gold_exact"] = g_exact
                row["pred_exact"] = p_exact
                row["gold_cmp"] = g_cmp
                row["pred_cmp"] = p_cmp
                row["is_correct"] = str(ok)
                row["judge_error"] = ""
            except Exception as e:
                # 読み変換などで例外が出ても落とさず行に残す
                row["gold"] = gold
                row["gold_exact"] = ""
                row["pred_exact"] = ""
                row["gold_cmp"] = ""
                row["pred_cmp"] = ""
                row["is_correct"] = "False"
                row["judge_error"] = f"{type(e).__name__}: {e}"

            w.writerow(row)

    print(f"[DONE] wrote: {out_path}")


if __name__ == "__main__":
    main()
