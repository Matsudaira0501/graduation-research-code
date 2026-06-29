import csv
import os
import re

# === 必要に応じて、実際の CSV のヘッダ名に合わせて変更してください ===
ID_COL = "id"           # 文書IDが入っている列名
KEYPHRASE_COL = "keyphrases"  # キーフレーズ群が入っている列名
# =================================================================


def load_stop_terms(path: str):
    """
    stop_terms.txt を読み込んでストップワード集合を返す。
    1行1語, UTF-8 を想定。
    ファイルが無ければ空集合を返す。
    """
    if not path or not os.path.exists(path):
        return set()

    stop = set()
    # stop_terms.txt は UTF-8 で保存している前提
    with open(path, encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if term:
                stop.add(term)
    return stop


def load_keyphrase_csv(path: str, stop_terms=None):
    """
    lv1_keyphrases_all.csv から

      - doc_ids     : 文書IDのリスト (str)
      - docs_terms  : 類似度計算用のキーフレーズ集合のリスト (set[str])
      - docs_texts  : 表示用の元テキスト（セル内改行を含む）リスト (str)

    を読み込む。

    前提:
      - lv1_keyphrases_all.csv は cp932（Shift_JIS）で保存されている
      - ID_COL / KEYPHRASE_COL に対応する列が存在する
      - KEYPHRASE_COL の中に「セル内改行付き」でキーフレーズが並んでいる
    """
    if stop_terms is None:
        stop_terms = set()

    doc_ids = []
    docs_terms = []
    docs_texts = []

    # Excel から保存した CSV を想定して cp932 で読む
    with open(path, newline="", encoding="cp932") as f:
        reader = csv.DictReader(f)

        # 必要な列が存在するか確認
        if ID_COL not in reader.fieldnames or KEYPHRASE_COL not in reader.fieldnames:
            raise KeyError(
                f"CSV に必要な列 '{ID_COL}' または '{KEYPHRASE_COL}' がありません。\n"
                f"fieldnames = {reader.fieldnames}"
            )

        for row in reader:
            if not row:
                continue

            doc_id = (row[ID_COL] or "").strip()
            text = row[KEYPHRASE_COL] or ""

            # そのまま出力に使う用のテキスト（セル内改行を保持）
            docs_texts.append(text)
            doc_ids.append(doc_id)

            # キーフレーズを分割
            # 改行, /, ／, 、, カンマ, セミコロン, パイプ などを区切りとする
            raw_phrases = re.split(r"[\n／/、,;|]+", text)
            phrases = [
                p.strip()
                for p in raw_phrases
                if p.strip()
            ]

            # ストップワードを除去して集合に
            terms = {p for p in phrases if p not in stop_terms}
            docs_terms.append(terms)

    return doc_ids, docs_terms, docs_texts


def jaccard(a: set, b: set) -> float:
    """
    Jaccard 係数 J(A,B) = |A∩B| / |A∪B|
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def compute_topk_neighbors(docs_terms, top_k=3, min_sim=0.0):
    """
    すべての文書をクエリとして、上位 top_k 件の近傍文書を Jaccard 類似度で求める。

    min_sim 以下の類似度は無視する（0 を除外したい場合は min_sim=0.0）。

    戻り値:
        { query_idx: [(neighbor_idx, sim), ...], ... }
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

            # スコア 0 は除外
            if sim <= min_sim:
                continue

            sims.append((o_idx, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        results[q_idx] = sims[:top_k]

    return results


def write_neighbors_to_csv(results, doc_ids, docs_texts, output_path, top_k=3):
    """
    類似度計算結果を CSV に出力する。

    出力形式:
        query,query_doc,
        neighbor1,neighbor1_doc,
        neighbor2,neighbor2_doc,
        neighbor3,neighbor3_doc

    - query / neighborX には doc_ids を使う。
    - neighborX には "doc_id,score(小数3桁)" を入れる。
    - *_doc には対応する文書のキーフレーズテキストをそのまま入れる。
    """
    n_docs = len(docs_texts)
    assert len(doc_ids) == n_docs

    # Excel で文字化けしにくいよう cp932 で出力
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
                    cell = f"{n_id},{sim:.3f}"  # "文書ID,0.123"
                    row.append(cell)
                    row.append(docs_texts[n_idx])
                else:
                    row.append("")
                    row.append("")

            writer.writerow(row)


def main():
    # 入力ファイル
    input_csv = "lv1_keyphrases_all.csv"
    stop_terms_path = "stop_terms.txt"   # 無ければ無視される
    output_csv = "jaccard_neighbors_from_keyphrases.csv"

    # 1) stop_terms 読み込み（存在しない場合は空集合）
    stop_terms = load_stop_terms(stop_terms_path)

    # 2) キーフレーズ CSV の読み込み（cp932）
    doc_ids, docs_terms, docs_texts = load_keyphrase_csv(
        input_csv, stop_terms=stop_terms
    )

    # 3) Jaccard 類似度で近傍計算（スコア 0 は除外）
    results = compute_topk_neighbors(docs_terms, top_k=5, min_sim=0.0)

    # 4) 結果を CSV 出力
    write_neighbors_to_csv(results, doc_ids, docs_texts, output_csv, top_k=5)


if __name__ == "__main__":
    main()
