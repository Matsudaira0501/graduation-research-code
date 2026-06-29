import csv
import re


def parse_lv1_file(input_path):
    """
    lv1_output.txt のような形式のファイルから、
    各ブロックを 1 文書として

      - キーワード集合 (docs_terms)
      - 文書テキスト（そのブロックの行をセル内改行でつなげたもの, docs_texts）

    を返す。

    Returns
    -------
    docs_terms : list[set[str]]
        docs_terms[i] が i 番目(0始まり)の文書のキーワード集合
    docs_texts : list[str]
        docs_texts[i] が i 番目(0始まり)の文書のテキスト（CSV 出力用, セル内改行含む）
    """
    docs_terms = []
    docs_texts = []

    current_terms = []
    current_lines = []

    # " 1. いんく  (score=0.211)" のような行から
    # キーワード部分だけを取り出す正規表現
    line_pattern = re.compile(r"^\s*\d+\.\s*(.*?)\s*\(score=")

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
                if current_terms:
                    # キーワード集合を保存
                    docs_terms.append(set(current_terms))
                    # 文書テキストを 1 つの文字列にまとめる（セル内改行）
                    doc_text = "\n".join(
                        l
                        for l in current_lines
                        if l.strip() and not l.strip().startswith("#")
                    )
                    docs_texts.append(doc_text)

                    current_terms = []
                    current_lines = []
                continue

            # ブロック内の通常行
            current_lines.append(line)

            # キーワード抽出
            m = line_pattern.match(line)
            if m:
                term = m.group(1).strip()
                if term:
                    current_terms.append(term)

    # ファイル末尾のブロックを反映
    if current_terms:
        docs_terms.append(set(current_terms))
        doc_text = "\n".join(
            l
            for l in current_lines
            if l.strip() and not l.strip().startswith("#")
        )
        docs_texts.append(doc_text)

    return docs_terms, docs_texts


def jaccard(set_a, set_b):
    """
    2つの集合の Jaccard 係数を計算する。
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set_a and not set_b:
        return 0.0

    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union


def compute_topk_neighbors(docs_terms, top_k=3):
    """
    すべての文書をクエリとして見たときの
    上位 top_k 近傍文書（Jaccard 類似度順）を計算する。

    Parameters
    ----------
    docs_terms : list[set[str]]
        docs_terms[i] が i 番目(0始まり)の文書のキーワード集合
    top_k : int
        何件の近傍を出力するか

    Returns
    -------
    dict[int, list[tuple[int, float]]]
        { query_internal_idx: [(neighbor_internal_idx, sim), ...], ... }
        ※ internal_idx は 0 始まりのインデックス
    """
    results = {}
    n_docs = len(docs_terms)

    for q_idx in range(n_docs):
        q_terms = docs_terms[q_idx]

        sims = []
        for o_idx in range(n_docs):
            if o_idx == q_idx:
                continue  # 自分自身は除外
            o_terms = docs_terms[o_idx]

            sim = jaccard(q_terms, o_terms)
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
    input_path = "lv1_output2.txt"          # 実際のファイル名に合わせて変更
    output_path = "jaccard_neighbors2.csv"  # 出力 CSV
    top_k = 3                              # 上位3件のみ

    # 1. lv1_output.txt から各文書のキーワード集合 & テキストを取得
    docs_terms, docs_texts = parse_lv1_file(input_path)

    # 2. 各文書をクエリとして、Jaccard による上位3件の近傍を計算（internal idx ベース）
    results = compute_topk_neighbors(docs_terms, top_k=top_k)

    # 3. 指定フォーマットで CSV 出力（外部表示IDは N..1 の逆順）
    write_neighbors_to_csv(results, docs_texts, output_path, top_k=top_k)


if __name__ == "__main__":
    main()
