import csv
import re


def parse_lv1_file(input_path):
    """
    lv1_output.txt のような形式のファイルから、
    各ブロックを 1 文書として

      - キーワード -> score の辞書 (docs_weights)
      - 文書テキスト（そのブロックの行をセル内改行でつなげたもの, docs_texts）

    を返す。

    Returns
    -------
    docs_weights : list[dict[str, float]]
        docs_weights[i][term] が i 番目(0始まり)の文書における term の重み(score)
    docs_texts : list[str]
        docs_texts[i] が i 番目(0始まり)の文書のテキスト（CSV 出力用, セル内改行含む）
    """
    docs_weights = []
    docs_texts = []

    current_weights = {}  # term -> score
    current_lines = []

    # " 1. いんく  (score=0.211)" のような行から
    # キーワードと score を両方取り出す正規表現
    line_pattern = re.compile(
        r"^\s*\d+\.\s*(.*?)\s*\(score=([0-9.eE+-]+)\)"
    )

    with open(input_path, encoding="utf-8") as f:
        for raw_line in f:
            # 行末の改行は削る（セル内改行は後で付ける）
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            # 空行は無視
            if not stripped:
                continue

            # ブロック区切り（# の行）
            if stripped.startswith("#"):
                if current_weights:
                    # 重み辞書を保存
                    docs_weights.append(dict(current_weights))
                    # 文書テキストを 1 つの文字列にまとめる（セル内改行）
                    doc_text = "\n".join(
                        l
                        for l in current_lines
                        if l.strip() and not l.strip().startswith("#")
                    )
                    docs_texts.append(doc_text)

                    current_weights = {}
                    current_lines = []
                continue

            # ブロック内の通常行
            current_lines.append(line)

            # キーワード & score 抽出
            m = line_pattern.match(line)
            if m:
                term = m.group(1).strip()
                score_str = m.group(2).strip()
                if term and score_str:
                    try:
                        score = float(score_str)
                    except ValueError:
                        score = 0.0
                    current_weights[term] = score

    # ファイル末尾のブロックを反映
    if current_weights:
        docs_weights.append(dict(current_weights))
        doc_text = "\n".join(
            l
            for l in current_lines
            if l.strip() and not l.strip().startswith("#")
        )
        docs_texts.append(doc_text)

    return docs_weights, docs_texts


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

    Parameters
    ----------
    docs_weights : list[dict[str, float]]
        docs_weights[i] が i 番目(0始まり)の文書の {term: score}
    top_k : int
        何件の近傍を出力するか

    Returns
    -------
    dict[int, list[tuple[int, float]]]
        { query_internal_idx: [(neighbor_internal_idx, sim), ...], ... }
        ※ internal_idx は 0 始まりのインデックス
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
            sims.append((o_idx, sim))

        # 類似度の高い順にソート
        sims.sort(key=lambda x: x[1], reverse=True)

        # 上位 top_k 件だけ残す
        results[q_idx] = sims[:top_k]

    return results


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
            # 外部表示用のクエリID（逆順）
            q_external_id = n_docs - q_idx
            # クエリ文書のテキスト
            q_text = docs_texts[q_idx]

            row = [q_external_id, q_text]

            for i in range(top_k):
                if i < len(neighbors):
                    n_idx, sim = neighbors[i]
                    # 外部表示用 neighbor doc_id（逆順）
                    n_external_id = n_docs - n_idx

                    # neighbor と sim を同じセルに格納
                    # 区切り文字はカンマ, 小数点以下3桁
                    cell = f"{n_external_id},{sim:.3f}"
                    row.append(cell)

                    # その neighbor 文書のテキスト（セル内改行付き）
                    doc_text = docs_texts[n_idx]  # docs_texts は internal_idx でアクセス
                    row.append(doc_text)
                else:
                    # 近傍が足りない場合は空欄で埋める
                    row.append("")
                    row.append("")
            writer.writerow(row)


def main():
    # 入力と出力のファイル名
    input_path = "lv1_output2.txt"                 # 実際のファイル名に合わせて変更
    output_path = "weighted_jaccard_neighbors2.csv"  # 出力 CSV（単純版と分けたいので別名に）
    top_k = 3                                     # 上位3件のみ

    # 1. lv1_output.txt から各文書の {term: score} & テキストを取得
    docs_weights, docs_texts = parse_lv1_file(input_path)

    # 2. 各文書をクエリとして、重み付き Jaccard による上位3件の近傍を計算
    results = compute_topk_neighbors(docs_weights, top_k=top_k)

    # 3. 指定フォーマットで CSV 出力（外部表示IDは N..1 の逆順）
    write_neighbors_to_csv(results, docs_texts, output_path, top_k=top_k)


if __name__ == "__main__":
    main()
