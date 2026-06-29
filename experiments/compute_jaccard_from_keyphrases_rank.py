import argparse
import itertools
from collections import defaultdict

import pandas as pd


def rank_to_weight(rank: int, k: int) -> float:
    """
    ランク r (1始まり) に対して w(r) = (k + 1 - r) / k の重みを返す。
    k はその文書の総キーフレーズ数。
    """
    return (k + 1 - rank) / k


def build_phrase_weights(phrases_in_rank_order):
    """
    ランク順に並んだキーフレーズのリストから
    { phrase: weight } の辞書を作る。
    """
    k = len(phrases_in_rank_order)
    weights = {}
    for idx, phrase in enumerate(phrases_in_rank_order):
        rank = idx + 1  # 1-based
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


def load_keyphrases(csv_path: str,
                    doc_col: str = "doc_id",
                    phrase_col: str = "keyphrase",
                    rank_col: str | None = None):
    """
    キーフレーズ CSV を読み込んで

    - doc_id -> [phrase1, phrase2, ...]（ランク順）の dict
    を返す。

    rank_col を指定した場合:
        同一 doc 内で rank_col の昇順でソートしてランク順とみなす。
    指定しない場合:
        ファイル中の出現順をそのままランク順とみなす。
    """
    df = pd.read_csv(csv_path)

    if rank_col is not None and rank_col in df.columns:
        df = df.sort_values(by=[doc_col, rank_col])

    docs = defaultdict(list)
    for _, row in df.iterrows():
        doc_id = row[doc_col]
        phrase = row[phrase_col]
        docs[doc_id].append(phrase)

    return dict(docs)


def compute_similarity_matrix(docs_phrases: dict) -> pd.DataFrame:
    """
    doc_id -> [phrase1, phrase2, ...] から
    Weighted Jaccard 類似度の行列 (DataFrame) を作成する。
    """
    doc_ids = list(docs_phrases.keys())

    # 先に各文書の {phrase: weight} を作っておく
    doc_weights = {
        doc_id: build_phrase_weights(phrases)
        for doc_id, phrases in docs_phrases.items()
    }

    # 類似度行列を埋める
    sim_df = pd.DataFrame(
        index=doc_ids,
        columns=doc_ids,
        dtype=float
    )

    for d1, d2 in itertools.combinations(doc_ids, 2):
        s = weighted_jaccard(doc_weights[d1], doc_weights[d2])
        sim_df.at[d1, d2] = s
        sim_df.at[d2, d1] = s

    # 対角成分は 1.0 にしておく
    for d in doc_ids:
        sim_df.at[d, d] = 1.0

    return sim_df


def compute_topk_neighbors(sim_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    類似度行列から各 doc ごとに上位 top_k の近傍 doc とスコアを出す。

    出力例:
        doc_id,neighbor1,sim1,neighbor2,sim2,...
    """
    rows = []
    for doc_id in sim_df.index:
        # 自分自身を除いて降順ソート
        sims = sim_df.loc[doc_id].drop(index=doc_id).sort_values(ascending=False)
        top = sims.head(top_k)

        row = {"doc_id": doc_id}
        for i, (nbr, score) in enumerate(top.items(), start=1):
            row[f"neighbor{i}"] = nbr
            row[f"sim{i}"] = score
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Rank-based weighted Jaccard similarity for keyphrases."
    )
    parser.add_argument("input_csv", help="キーフレーズ一覧 CSV のパス")
    parser.add_argument(
        "--doc-col", default="doc_id",
        help="文書 ID の列名 (default: doc_id)"
    )
    parser.add_argument(
        "--phrase-col", default="keyphrase",
        help="キーフレーズ列名 (default: keyphrase)"
    )
    parser.add_argument(
        "--rank-col", default=None,
        help="ランクを表す列名。指定しない場合は出現順をランクとみなす。"
    )
    parser.add_argument(
        "--topk", type=int, default=5,
        help="近傍として出力する文書数 (default: 5)"
    )
    parser.add_argument(
        "--sim-matrix-out", default="weighted_sim_matrix.csv",
        help="類似度行列を書き出す CSV パス"
    )
    parser.add_argument(
        "--neighbors-out", default="weighted_neighbors.csv",
        help="近傍一覧を書き出す CSV パス"
    )

    args = parser.parse_args()

    # 1. キーフレーズ読込
    docs_phrases = load_keyphrases(
        csv_path=args.input_csv,
        doc_col=args.doc_col,
        phrase_col=args.phrase_col,
        rank_col=args.rank_col,
    )

    # 2. Weighted Jaccard 類似度行列
    sim_df = compute_similarity_matrix(docs_phrases)
    sim_df.to_csv(args.sim_matrix_out, index=True)

    # 3. 各文書の top-k 近傍
    neighbors_df = compute_topk_neighbors(sim_df, top_k=args.topk)
    neighbors_df.to_csv(args.neighbors_out, index=False)

    print(f"[INFO] 類似度行列: {args.sim_matrix_out}")
    print(f"[INFO] 近傍一覧:   {args.neighbors_out}")


if __name__ == "__main__":
    main()
