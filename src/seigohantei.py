import argparse
import csv
import re
import unicodedata
from pathlib import Path

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    return s

def read_first_column_csv(path: str):
    """
    1列リストCSVを読む（ヘッダ無し想定）。
    末尾カンマ等で空列があっても、先頭列だけ使う。
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                rows.append("")
            else:
                rows.append(row[0])
    return rows

def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions against gold answers (row-aligned) without pandas.")
    ap.add_argument("--gold", required=True, help="正解CSV（1列リスト）")
    ap.add_argument("--pred", required=True, help="予測CSV（1列リスト）")
    ap.add_argument("--out", default="eval_detail.csv", help="判定結果の出力CSV")
    args = ap.parse_args()

    gold = read_first_column_csv(args.gold)
    pred = read_first_column_csv(args.pred)

    n_gold, n_pred = len(gold), len(pred)
    n = min(n_gold, n_pred)
    if n_gold != n_pred:
        print(f"[WARN] row count mismatch: gold={n_gold}, pred={n_pred}. Evaluate first {n} rows only.")

    correct = 0
    out_path = Path(args.out).resolve()

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gold", "pred", "gold_norm", "pred_norm", "is_correct"])

        for i in range(n):
            g = gold[i]
            p = pred[i]
            g_norm = normalize_answer(g)
            p_norm = normalize_answer(p)
            is_ok = (g_norm == p_norm)
            if is_ok:
                correct += 1
            writer.writerow([i + 1, g, p, g_norm, p_norm, str(is_ok)])

        total = n
        acc = (correct / total) if total > 0 else 0.0

        # 1行空けてからサマリー行を追加（Excelで見やすく）
        writer.writerow([])
        writer.writerow(["SUMMARY", "", "", "", "", ""])
        writer.writerow(["accuracy", acc, "correct", correct, "total", total])

    print(f"[RESULT] accuracy = {acc:.4f} ({correct}/{total})")
    print(f"[DONE] wrote: {out_path}")

if __name__ == "__main__":
    main()
