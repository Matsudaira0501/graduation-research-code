import argparse
import itertools
import re

import pandas as pd


def parse_keyphrase_block(text: str) -> list[str]:
    """
    '1. たいよう\\n2. ちきゅう\\n...' のような文字列を
    ランク順のキーフレーズリストに変換する。
    """
    if pd.isna(text):
        return []

    phrases = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        # 先頭の「数字. 」を削除（例: "1. たいよう" -> "たいよう"）
        line = re.sub(r'^\s*\d+\.\s*', '', line)
        if line:
            phrases.append(line)
    return phrases


def rank_to_weight(rank: int, k: int) -> float:
    """
    ランク r (1始まり) に対して重み w(r) = (k + 1 - r) / k を返す。
    k はその文書の総キーフレーズ数。
    例: k=20 のとき
      r=1  -> 1.0
      r=2  -> 0.95
      ...
      r=20 -> 0.05
    """
    return (k + 1 - rank) / k


def build_phrase_weights(phrases_in_rank_order: list[str]) -> dict[str, float]:
    """
    ランク順に並んだキーフレーズのリストから {phrase: weight} の辞書を作る。
    """
    k = len(phrases_in_rank_order)
    if k == 0:
        return {}

    weights = {}
    for idx, phrase in enumerate(phrases_in_rank_order):
        rank = idx + 1  # 1-based rank
        weights[phrase] = rank_to_weight(rank, k)
    return weights


def weighted_jaccard(weights_a: dict, weights_b: dict) -> float:
    """
    2 文書の {phrase: weight} 辞書から Weighted Jaccard 類似度を計算する。

    sim(A,B) = sum_p min(wA(p), wB(p)) / sum_p max(wA(p), wB(p))
    """
    all_phrases = set(weights_a.keys()) | set(weights_b.keys())
    num = 0.0
    den = 0.0

    for p in all_phrases:
        wa = weights_a.get(p, 0.0)
        wb = weights_b.get(p, 0.0)
        num += min(wa, wb)
        den += max(wa, wb)

    if den == 0.0:
        return 0.0
    return num / den


def load_docs(csv_path: str, encoding: str = "cp932"):
    """
    lv1_keyphrases_all.csv を読み込み、
    - id -> keyphrases（元テキスト）
    - id -> [phrase1, phrase2, ...]（ランク順）
    を返す。
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    id_to_text = {}
    id_to_phrases = {}

    for _, row in df.iterrows():
        doc_id = int(row["id"])
        kp_text = row["keyphrases"]
        id_to_text[doc_id] = kp_text
        id_to_phrases[doc_id] = parse_keyphrase_block(kp_text)

    return id_to_text, id_to_phrases


def compute_similarity_matrix(id_to_phrases: dict[int, list[str]]) -> pd.DataFrame:
    """
    id -> [phrase1, phrase2, ...] から Weighted Jaccard 類似度行列を作成。
    """
    doc_ids = sorted(id_to_phrases.keys())

    # 各文書の {phrase: weight} を作成
    id_to_weights = {
        doc_id: build_phrase_weights(phrases)
        for doc_id, phrases in id_to_phrases.items()
    }

    sim_df = pd.DataFrame(index=doc_ids, columns=doc_ids, dtype=float)

    # 対角以外を計算
    for i, d1 in enumerate(doc_ids):
        w1 = id_to_weights[d1]
        for d2 in doc_ids[i + 1 :]:
            w2 = id_to_weights[d2]
            s = weighted_jaccard(w1, w2)
            sim_df.at[d1, d2] = s
            sim_df.at[d2, d1] = s

    # 対角成分は 1.0
    for d in doc_ids:
        sim_df.at[d, d] = 1.0

    return sim_df


def build_neighbors_df(
    sim_df: pd.DataFrame,
    id_to_text: dict[int, str],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Jaccard 類似度行列から
    query, query_doc, neighbor1, neighbor1_doc, ..., neighbor5, neighbor5_doc
    形式の DataFrame を作成する。
    """
    rows = []
    for qid in sim_df.index:
        qtext = id_to_text[int(qid)]

        # 自分自身を除いて降順ソート
        sims = sim_df.loc[qid].drop(index=qid).dropna()
        # 類似度 0 は出力しない場合
        sims = sims[sims > 0.0]
        sims = sims.sort_values(ascending=False)

        row = {"query": int(qid), "query_doc": qtext}

        if len(sims) == 0:
            # 近傍がない場合は neighbor 列は NaN のまま
            rows.append(row)
            continue

        # 上位 top_k を取り出して "id,score" 形式で埋める
        for i, (nid, score) in enumerate(sims.head(top_k).items(), start=1):
            nid_int = int(nid)
            row[f"neighbor{i}"] = f"{nid_int},{score:.3f}"
            row[f"neighbor{i}_doc"] = id_to_text[nid_int]

        rows.append(row)

    # 列順を jaccard_neighbors_from_keyphrases.csv と同じにする
    cols = ["query", "query_doc"]
    for i in range(1, top_k + 1):
        cols.append(f"neighbor{i}")
        cols.append(f"neighbor{i}_doc")

    df_out = pd.DataFrame(rows)
    df_out = df_out.reindex(columns=cols)

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Rank-weighted (w(r)=(k+1-r)/k) Jaccard neighbors from keyphrases."
    )
    parser.add_argument(
        "--input",
        default="lv1_keyphrases_all.csv",
        help="入力 CSV ファイル (default: lv1_keyphrases_all.csv)",
    )
    parser.add_argument(
        "--output",
        default="rank_weighted_neighbors_from_keyphrases.csv",
        help="出力 CSV ファイル (default: rank_weighted_neighbors_from_keyphrases.csv)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="各 query ごとに出力する近傍数 (default: 5)",
    )

    args = parser.parse_args()

    # 1. 文書読み込み
    id_to_text, id_to_phrases = load_docs(args.input)

    # 2. Weighted Jaccard 類似度行列
    sim_df = compute_similarity_matrix(id_to_phrases)

    # 3. 近傍リストに変換（出力形式を jaccard_neighbors_from_keyphrases.csv に合わせる）
    neighbors_df = build_neighbors_df(sim_df, id_to_text, top_k=args.topk)

    # 4. 保存（日本語を含むので BOM 付き UTF-8 にしておく）
    neighbors_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"[INFO] 出力を書き出しました: {args.output}")


if __name__ == "__main__":
    main()
