import csv
import re


# ===== ストップワード読み込み =====

def load_stop_terms(path):
    """
    1行1語の stop_terms.txt からストップワード集合を読み込む。
    """
    stop = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if term:
                stop.add(term)
    return stop


# ===== 入力ファイルのパース =====

def parse_lv1_file(input_path, stop_terms=None):
    """
    lv1_output.txt から、
      - 各文書の {term: score} 辞書 (docs_weights)
      - 各文書のテキスト (docs_texts; セル内改行用の文字列)
    を返す。

    stop_terms に含まれる term は {term: score} から除外するが、
    docs_texts には元の行をそのまま残す。
    """
    if stop_terms is None:
        stop_terms = set()

    docs_weights = []
    docs_texts = []

    current_weights = {}
    current_lines = []

    # 行の形式: " 1. いんく  (score=0.211)" など
    line_pattern = re.compile(
        r"^\s*\d+\.\s*(.*?)\s*\(score=([0-9.eE+-]+)\)"
    )

    with open(input_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            # ブロック区切り（#）
            if stripped.startswith("#"):
                if current_weights:
                    docs_weights.append(dict(current_weights))
                    doc_text = "\n".join(
                        l
                        for l in current_lines
                        if l.strip() and not l.strip().startswith("#")
                    )
                    docs_texts.append(doc_text)

                    current_weights = {}
                    current_lines = []
                continue

            # ブロック内行
            current_lines.append(line)

            m = line_pattern.match(line)
            if m:
                term = m.group(1).strip()
                score_str = m.group(2).strip()
                if not term or not score_str:
                    continue

                # ストップワードは類似度計算から除外
                if term in stop_terms:
                    continue

                try:
                    score = float(score_str)
                except ValueError:
                    score = 0.0

                current_weights[term] = score

    # ファイル末尾のブロック
    if current_weights:
        docs_weights.append(dict(current_weights))
        doc_text = "\n".join(
            l
            for l in current_lines
            if l.strip() and not l.strip().startswith("#")
        )
        docs_texts.append(doc_text)

    return docs_weights, docs_texts


# ===== 重み付き Jaccard =====

def weighted_jaccard(weights_a, weights_b):
    """
    重み付き Jaccard 係数を計算する。
    weights_a, weights_b は {term: score} の辞書。

    J_w(A,B) = sum_i min(a_i, b_i) / sum_i max(a_i, b_i)
    """
    terms = set(weights_a.keys()) | set(weights_b.keys())
    if not terms:
        return 0.0

    num = 0.0
    den = 0.0

    for t in terms:
        a = weights_a.get(t, 0.0)
        b = weights_b.get(t, 0.0)
        num += min(a, b)
        den += max(a, b)

    if den == 0.0:
        return 0.0

    return num / den


def compute_topk_neighbors(docs_weights, top_k=3):
    """
    すべての文書をクエリとして見たときの
    上位 top_k 近傍文書（重み付き Jaccard 類似度順）を計算する。

    ※ 類似度が 0 のものは neighbor から除外する。
    """
    results = {}
    n_docs = len(docs_weights)

    for q_idx in range(n_docs):
        q_w = docs_weights[q_idx]

        sims = []
        for o_idx in range(n_docs):
            if o_idx == q_idx:
                continue  # 自分自身は除外
            o_w = docs_weights[o_idx]

            sim = weighted_jaccard(q_w, o_w)
            # ★ スコア 0 は除外
            if sim <= 0.0:
                continue

            sims.append((o_idx, sim))

        # 類似度の高い順にソート
        sims.sort(key=lambda x: x[1], reverse=True)

        # 上位 top_k 件だけ残す
        results[q_idx] = sims[:top_k]

    return results


# ===== CSV 出力 =====

def write_neighbors_to_csv(results, docs_texts, output_path, top_k=3):
    """
    {query_internal_idx: [(neighbor_internal_idx, sim), ...]} を
    1 行 1 クエリ形式の CSV に書き出す。

    出力フォーマット:
        query,query_doc,
        neighbor1,neighbor1_doc,
        neighbor2,neighbor2_doc,
        neighbor3,neighbor3_doc

    - 文書番号は 1..N ではなく N..1 の逆順で付与。
      外部表示の doc_id = N - internal_idx
    - neighborX には "doc_id,score"（score は小数点以下3桁）を入れる。
      例: "40,0.823"
    - query_doc / neighborX_doc には、その文書のテキスト（複数行）を
      セル内改行（\n）付きで格納する。
    """
    n_docs = len(docs_texts)

    # Excel で文字化けしないように cp932 で保存
    with open(output_path, "w", newline="", encoding="cp932") as f:
        writer = csv.writer(f)

        # ヘッダ
        header = ["query", "query_doc"]
        for i in range(1, top_k + 1):
            header.append(f"neighbor{i}")
            header.append(f"neighbor{i}_doc")
        writer.writerow(header)

        # internal_idx の昇順で回して、外部IDは逆順を付ける
        for q_idx, neighbors in sorted(results.items(), key=lambda x: x[0]):
            q_external_id = n_docs - q_idx  # 逆順ID
            q_text = docs_texts[q_idx]

            row = [q_external_id, q_text]

            for i in range(top_k):
                if i < len(neighbors):
                    n_idx, sim = neighbors[i]
                    n_external_id = n_docs - n_idx

                    cell = f"{n_external_id},{sim:.3f}"
                    row.append(cell)

                    doc_text = docs_texts[n_idx]
                    row.append(doc_text)
                else:
                    # スコア > 0 の neighbor が top_k 未満なら空欄
                    row.append("")
                    row.append("")
            writer.writerow(row)


# ===== メイン関数 =====

def main():
    input_path = "lv1_output2.txt"                # 元データ
    stop_path = "stop_terms.txt"                 # ストップワード
    output_path = "weighted_jaccard_neighbors_nostop2.csv"
    top_k = 3

    # 1) ストップワード読み込み
    stop_terms = load_stop_terms(stop_path)

    # 2) lv1_output.txt をパース（ストップワード除外）
    docs_weights, docs_texts = parse_lv1_file(input_path, stop_terms=stop_terms)

    # 3) 重み付き Jaccard で近傍計算（スコア 0 は除外）
    results = compute_topk_neighbors(docs_weights, top_k=top_k)

    # 4) CSV 出力
    write_neighbors_to_csv(results, docs_texts, output_path, top_k=top_k)


if __name__ == "__main__":
    main()
