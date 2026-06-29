# ファイル名例: llm_keywords_jaccard.py
import csv


def load_stop_terms(path):
    """
    stop_terms.txt を読み込んでストップワード集合を返す。
    1行1語、UTF-8 を想定。
    """
    stop = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if term:
                stop.add(term)
    return stop


def load_llm_keywords(path, stop_terms=None):
    """
    ChatGPT.txt から文書ごとのキーフレーズ集合と元テキストを読み込む。

    前提:
        - 1行が1文書
        - その1行の中で、改行区切りでキーフレーズが並んでいる
          （CSVの1セル内に改行入りで保存されているイメージ）

    戻り値:
        docs_terms : list[set[str]]   # 類似度計算用（ストップワード除去済み）
        docs_texts : list[str]        # 表示用の元テキスト（セル内改行付き）
    """
    if stop_terms is None:
        stop_terms = set()

    docs_terms = []
    docs_texts = []

    # ChatGPT.txt は 1列だけのCSVという想定で読む
    with open(path, encoding="cp932", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            text = row[0]
            docs_texts.append(text)

            # 1文書内のキーフレーズを改行で区切る
            phrases = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            # ストップワードを除外
            filtered = [
                p for p in phrases
                if p not in stop_terms
            ]

            docs_terms.append(set(filtered))

    return docs_terms, docs_texts


def jaccard(set_a, set_b):
    """
    普通の Jaccard 係数を計算する。
        J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set_a and not set_b:
        return 0.0

    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union


def compute_topk_neighbors(docs_terms, top_k=3, min_sim=0.0):
    """
    すべての文書をクエリとして、上位 top_k 件の近傍文書を Jaccard 類似度で求める。

    min_sim で、閾値未満の類似度は無視する（0 を除外したい場合など）。
    """
    results = {}
    n_docs = len(docs_terms)

    for q_idx in range(n_docs):
        q_terms = docs_terms[q_idx]
        sims = []

        for o_idx in range(n_docs):
            if o_idx == q_idx:
                continue

            o_terms = docs_terms[o_idx]
            sim = jaccard(q_terms, o_terms)

            # 閾値以下はスキップ
            if sim <= min_sim:
                continue

            sims.append((o_idx, sim))

        # 類似度の高い順にソート
        sims.sort(key=lambda x: x[1], reverse=True)

        # 上位 top_k 件だけ残す
        results[q_idx] = sims[:top_k]

    return results


def write_neighbors_to_csv(results, docs_texts, doc_ids, output_path, top_k=3):
    """
    類似度計算結果を CSV に出力する。

    出力形式:
        query,query_doc,
        neighbor1,neighbor1_doc,
        neighbor2,neighbor2_doc,
        neighbor3,neighbor3_doc

    - query, neighborX には元の文書ID（doc_ids の値）を使う。
    - neighborX のセルには "doc_id,score"（score は小数点以下3桁）。
    - *_doc には対応する文書のテキスト（セル内改行付き）。
    """
    n_docs = len(docs_texts)
    assert len(doc_ids) == n_docs

    # Excel で文字化けしないよう cp932 で保存
    with open(output_path, "w", newline="", encoding="cp932") as f:
        writer = csv.writer(f)

        header = ["query", "query_doc"]
        for i in range(1, top_k + 1):
            header.append(f"neighbor{i}")
            header.append(f"neighbor{i}_doc")
        writer.writerow(header)

        for q_idx, neighbors in sorted(results.items(), key=lambda x: x[0]):
            q_id = doc_ids[q_idx]
            q_text = docs_texts[q_idx]

            row = [q_id, q_text]

            for i in range(top_k):
                if i < len(neighbors):
                    n_idx, sim = neighbors[i]
                    n_id = doc_ids[n_idx]
                    cell = f"{n_id},{sim:.3f}"
                    row.append(cell)
                    row.append(docs_texts[n_idx])
                else:
                    row.append("")
                    row.append("")

            writer.writerow(row)


def main():
    # 入力ファイル
    input_path = "ChatGPT.txt"      # LLMで抽出したキーフレーズ
    stop_path = "stop_terms.txt"    # 既存のストップワード
    output_path = "llm_jaccard_neighbors.csv"

    # 文書ID（ここでは 50〜61 として対応付け）
    doc_ids = list(range(50, 62))   # [50, 51, ..., 61]

    # 1) ストップワード読み込み
    stop_terms = load_stop_terms(stop_path)

    # 2) ChatGPT.txt からキーフレーズ集合とテキストを読み込み
    docs_terms, docs_texts = load_llm_keywords(input_path, stop_terms=stop_terms)

    # 安全チェック（行数と doc_ids の長さが一致しているか）
    if len(docs_terms) != len(doc_ids):
        raise ValueError(f"文書数 {len(docs_terms)} と doc_ids の数 {len(doc_ids)} が一致しません")

    # 3) Jaccard 類似度で近傍計算（スコア0は除外したいので min_sim=0.0）
    results = compute_topk_neighbors(docs_terms, top_k=3, min_sim=0.0)

    # 4) CSV 出力
    write_neighbors_to_csv(results, docs_texts, doc_ids, output_path, top_k=3)


if __name__ == "__main__":
    main()
