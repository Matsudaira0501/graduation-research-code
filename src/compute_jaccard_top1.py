import argparse
import csv
import io
import re
import unicodedata
from typing import List, Tuple, Optional


# -----------------------------
# Utilities: encoding-safe read
# -----------------------------
def read_text_auto_encoding(path: str) -> str:
    """
    CSVファイルを（UTF-8-SIG/UTF-8/CP932/Shift_JIS の順で）自動判定して読み込む。
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error or RuntimeError("Failed to read file with known encodings.")


def split_csv_rows(text: str) -> List[List[str]]:
    """
    セル内改行（クォートで保護された改行）を正しく扱うため、
    splitlines() は使わず StringIO + csv.reader で読む。
    """
    f = io.StringIO(text)
    reader = csv.reader(f)
    return [row for row in reader]


# -----------------------------
# Parsing: CSV cell -> terms
# -----------------------------
# 例:
#  "1. 九九表"
#  "1. 九九表 (score=0.211)"
#  "1) 九九表"
#  "1）九九表"
#  を想定して広めに許容する
_KEY_LINE_RE = re.compile(
    r"^\s*\d+\s*[\.\)\uff09]\s*(.+?)\s*(?:\(\s*score\s*=\s*.*\))?\s*$",
    re.IGNORECASE,
)


def normalize_term(s: str) -> str:
    """最低限の表記ゆれ対策（NFKC + strip）"""
    s = unicodedata.normalize("NFKC", s)
    return s.strip()


def extract_terms_from_cell(cell: str) -> List[str]:
    """
    1セル内の複数行（キーフレーズ列）から term を抽出する。
    - 番号付き行（1. ... / 1) ...）はそれを優先
    - それ以外の行はフォールバックとして、その行自体を term として採用（空行以外）
      ※ノイズが混ざる入力の場合は、このフォールバックを切るのが良い
    """
    terms: List[str] = []
    if cell is None:
        return terms

    for line in str(cell).splitlines():
        line = line.strip()
        if not line:
            continue

        m = _KEY_LINE_RE.match(line)
        if m:
            term = normalize_term(m.group(1))
            if term:
                terms.append(term)
        else:
            # フォールバック：番号形式でない場合も拾う（llm-key以外の派生形式に対応）
            term = normalize_term(line)
            if term:
                terms.append(term)

    return terms


def parse_input_csv(path: str, column_index: int = 0) -> Tuple[List[str], List[set], List[str]]:
    """
    入力CSVを読み、各行を1文書として
      - doc_ids: 文書ID（"1".."N"）
      - docs_terms: term集合
      - docs_text: 元セル文字列
    を返す。

    既定では「各行の第1列（column_index=0）」をキーフレーズ欄として読む。
    """
    raw = read_text_auto_encoding(path)
    rows = split_csv_rows(raw)

    # 空行除去
    rows = [r for r in rows if any(c.strip() for c in r)]

    docs_text: List[str] = []
    docs_terms: List[set] = []

    for row in rows:
        cell = row[column_index] if len(row) > column_index else ""
        docs_text.append(cell)
        docs_terms.append(set(extract_terms_from_cell(cell)))

    doc_ids = [str(i + 1) for i in range(len(docs_text))]
    return doc_ids, docs_terms, docs_text


# -----------------------------
# Jaccard
# -----------------------------
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compute_top1_ties_aggregated(
    docs_terms: List[set],
    docs_text: List[str],
    reverse_id: bool = False,      # デフォルトは 1..N（= no-reverse）
    blank_if_zero: bool = True,
    tie_tol: float = 1e-12,
) -> List[Tuple[str, str, str, str]]:
    """
    各文書 i について、最大Jaccardの近傍を「同率も含めて全件」求め、
    1行にまとめて返す。

    出力:
      query, query_doc, top1_neighbors, top1_neighbor_docs

    - top1_neighbors: "id,score;id,score;..."
    - top1_neighbor_docs: "doc|||doc|||..."  (||| で連結)
    """
    n = len(docs_terms)
    out_ids = [str(n - i) for i in range(n)] if reverse_id else [str(i + 1) for i in range(n)]

    out_rows: List[Tuple[str, str, str, str]] = []

    for i in range(n):
        scores = [0.0] * n
        max_score = -1.0

        for j in range(n):
            if i == j:
                continue
            s = jaccard(docs_terms[i], docs_terms[j])
            scores[j] = s
            if s > max_score:
                max_score = s

        query_id = out_ids[i]
        query_doc = docs_text[i]

        if blank_if_zero and max_score <= 0.0:
            out_rows.append((query_id, query_doc, "", ""))
            continue

        pairs = []
        neighbor_docs = []
        for j in range(n):
            if i == j:
                continue
            if abs(scores[j] - max_score) <= tie_tol:
                pairs.append(f"{out_ids[j]},{max_score:.3f}")
                neighbor_docs.append(docs_text[j])

        out_rows.append(
            (query_id, query_doc, ";".join(pairs), "|||".join(neighbor_docs))
        )

    return out_rows


# -----------------------------
# I/O
# -----------------------------
def write_results_csv(out_path: str, rows: List[Tuple[str, str, str, str]]) -> None:
    header = ["query", "query_doc", "top1_neighbors", "top1_neighbor_docs"]
    with open(out_path, "w", encoding="cp932", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute Jaccard similarity and write ALL top-1 ties per query in a single row."
    )
    ap.add_argument("-i", "--input", required=True, help="Input CSV path (pke-key.csv / llm-key.csv).")
    ap.add_argument("-o", "--output", required=True, help="Output CSV path.")
    ap.add_argument(
        "--column",
        type=int,
        default=0,
        help="Which column contains the keyphrase block (default: 0).",
    )

    # デフォルトは 1..N。逆順が必要なときだけ --reverse-id
    ap.add_argument(
        "--reverse-id",
        action="store_true",
        help="If set, output IDs as N..1 (default is 1..N).",
    )
    # 互換用（付けてもデフォルトと同じ）
    ap.add_argument(
        "--no-reverse-id",
        action="store_true",
        help="(Compatibility) Default behavior is already no-reverse (1..N).",
    )

    ap.add_argument(
        "--allow-zero",
        action="store_true",
        help="If set, even when max Jaccard is 0.0, still output ties (may be many).",
    )
    ap.add_argument(
        "--tie-tol",
        type=float,
        default=1e-12,
        help="Tolerance for float tie judgment (default: 1e-12).",
    )
    ap.add_argument("--debug", action="store_true", help="Print debug info.")
    args = ap.parse_args()

    _, docs_terms, docs_text = parse_input_csv(args.input, column_index=args.column)

    if args.debug:
        print("[DEBUG] documents =", len(docs_terms))
        print("[DEBUG] terms per doc (first 10) =", [len(s) for s in docs_terms[:10]])

    rows = compute_top1_ties_aggregated(
        docs_terms=docs_terms,
        docs_text=docs_text,
        reverse_id=args.reverse_id,
        blank_if_zero=(not args.allow_zero),
        tie_tol=args.tie_tol,
    )
    write_results_csv(args.output, rows)


if __name__ == "__main__":
    main()
