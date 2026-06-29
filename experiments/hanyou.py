# ファイル名例: make_stopword_candidates.py
import re
from collections import defaultdict
import csv


def parse_lv1_terms_per_doc(input_path):
    """
    lv1_output.txt から、文書ごとの term 集合を返す。
    重み(score)は無視して、どの term がその文書に出ているかだけを見る。
    """
    docs_terms = []

    current_terms = set()
    # "1. いんく (score=0.211)" から term と score を抜くパターン
    line_pattern = re.compile(r"^\s*\d+\.\s*(.*?)\s*\(score=([0-9.eE+-]+)\)")

    with open(input_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            # ブロック区切り
            if stripped.startswith("#"):
                if current_terms:
                    docs_terms.append(current_terms)
                    current_terms = set()
                continue

            m = line_pattern.match(line)
            if m:
                term = m.group(1).strip()
                if term:
                    current_terms.add(term)

    # 最後のブロック
    if current_terms:
        docs_terms.append(current_terms)

    return docs_terms


def main():
    input_path = "lv1_output2.txt"
    output_path = "term_docfreq2.csv"

    docs_terms = parse_lv1_terms_per_doc(input_path)
    n_docs = len(docs_terms)

    # term が「何文書に出たか」のドキュメント頻度を数える
    docfreq = defaultdict(int)
    for terms in docs_terms:
        for t in terms:
            docfreq[t] += 1

    # CSV で出力: term, doc_freq, doc_ratio
    with open(output_path, "w", newline="", encoding="cp932") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "doc_freq", "doc_ratio"])
        for term, df in sorted(docfreq.items(), key=lambda x: x[1], reverse=True):
            ratio = df / n_docs
            writer.writerow([term, df, f"{ratio:.3f}"])


if __name__ == "__main__":
    main()
