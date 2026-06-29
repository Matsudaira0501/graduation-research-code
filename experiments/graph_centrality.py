import os

import numpy as np
import pandas as pd
import networkx as nx


def load_similarity_csv(csv_path):
    """
    類似度 CSV を読み込み、neighbor列・sim列を自動検出する。
    必須列: 'query', 'neighbor1'..., 'sim1'...
    """
    df = pd.read_csv(csv_path)

    # neighbor列とsim列を番号順に並べる
    neighbor_cols = [c for c in df.columns if c.startswith("neighbor")]
    sim_cols = [c for c in df.columns if c.startswith("sim")]

    neighbor_cols = sorted(neighbor_cols, key=lambda x: int(x.replace("neighbor", "")))
    sim_cols = sorted(sim_cols, key=lambda x: int(x.replace("sim", "")))

    if not neighbor_cols or not sim_cols:
        raise ValueError("neighbor列またはsim列が見つかりません。CSVの列名を確認してください。")

    if len(neighbor_cols) != len(sim_cols):
        raise ValueError("neighbor列とsim列の数が一致していません。")

    return df, neighbor_cols, sim_cols


def build_directed_graph(df, neighbor_cols, sim_cols):
    """
    CSV から有向グラフ G を構築する。
    ノード: query と neighbor のID（int）
    辺: query -> neighbor （重み = sim）
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        q = int(row["query"])
        G.add_node(q)

        for ncol, scol in zip(neighbor_cols, sim_cols):
            nid = row[ncol]
            sim = row[scol]

            if pd.isna(nid) or pd.isna(sim):
                continue

            nid = int(nid)
            sim = float(sim)

            # ノードを追加
            G.add_node(nid)
            # 有向辺 q -> nid （重み = 類似度）
            G.add_edge(q, nid, weight=sim)

    return G


def pagerank_power(G, alpha=0.85, max_iter=100, tol=1.0e-6, weight="weight"):
    """
    SciPy や networkx.pagerank_numpy を使わずに、
    単純なパワーイテレーションで PageRank を計算する。

    G: networkx.DiGraph
    alpha: ダンピング因子
    weight: エッジ重みの属性名
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    index = {node: i for i, node in enumerate(nodes)}

    # 初期ベクトル: 一様分布
    p = np.full(n, 1.0 / n, dtype=float)

    # あらかじめ out_weight_sum を計算しておく
    out_weight_sum = {}
    for node in nodes:
        total = 0.0
        for _, data in G[node].items():
            total += float(data.get(weight, 1.0))
        out_weight_sum[node] = total

    for _ in range(max_iter):
        p_new = np.zeros(n, dtype=float)

        # ランダムウォーク部分
        for i, node in enumerate(nodes):
            w_sum = out_weight_sum[node]
            if w_sum == 0.0:
                # ダングリングノード: 全ノードに均等にばらまく
                p_new += alpha * p[i] / n
            else:
                for nbr, data in G[node].items():
                    w = float(data.get(weight, 1.0))
                    j = index[nbr]
                    p_new[j] += alpha * p[i] * (w / w_sum)

        # テレポート（全ノード一様）分
        p_new += (1.0 - alpha) / n

        # 収束チェック（L1ノルム）
        if np.linalg.norm(p_new - p, 1) < tol:
            p = p_new
            break

        p = p_new

    # ノード -> スコア の dict に戻す
    return {node: float(p[index[node]]) for node in nodes}


def compute_centralities(G):
    """
    有向グラフ G から複数の中心性指標を計算し、DataFrame で返す。
    - in_degree（次数）
    - weighted_in_degree（重み付き）
    - out_degree / weighted_out_degree
    - PageRank（自前実装）
    - eigenvector centrality（無向グラフに変換して計算）
    """
    nodes = sorted(G.nodes())

    # 1. in-degree / out-degree（次数）
    in_deg = dict(G.in_degree())  # 重みなし
    out_deg = dict(G.out_degree())

    # 2. weighted in-degree / out-degree
    in_deg_w = dict(G.in_degree(weight="weight"))
    out_deg_w = dict(G.out_degree(weight="weight"))

    # 3. PageRank（重み付き） - 自前の power iteration 実装
    pr = pagerank_power(G, alpha=0.85, max_iter=200, tol=1.0e-8, weight="weight")

    # 4. Eigenvector centrality（無向に変換して重み付きで計算）
    Gu = G.to_undirected()
    try:
        evc = nx.eigenvector_centrality(Gu, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("[WARN] 固有ベクトル中心性が収束しませんでした。重みなしで再計算します。")
        evc = nx.eigenvector_centrality(Gu, max_iter=1000)

    # DataFrame にまとめる
    rows = []
    for node in nodes:
        row = {
            "node": node,
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "weighted_in_degree": in_deg_w.get(node, 0.0),
            "weighted_out_degree": out_deg_w.get(node, 0.0),
            "pagerank": pr.get(node, 0.0),
            "eigenvector_centrality": evc.get(node, 0.0),
        }
        rows.append(row)

    df_cent = pd.DataFrame(rows)
    return df_cent


def process_one_file(script_dir, input_name):
    """
    指定されたファイル名（similar_*.csv）について中心性を計算し、
    centrality_*.csv を出力する。
    """
    csv_path = os.path.join(script_dir, input_name)
    if not os.path.exists(csv_path):
        print(f"[INFO] スキップ: {input_name} は存在しません。")
        return

    print("=" * 80)
    print(f"[INFO] 読み込みCSV: {csv_path}")

    df, neighbor_cols, sim_cols = load_similarity_csv(csv_path)

    print(f"[INFO] neighbor列: {neighbor_cols}")
    print(f"[INFO] sim列     : {sim_cols}")

    G = build_directed_graph(df, neighbor_cols, sim_cols)
    print(f"[INFO] ノード数: {G.number_of_nodes()}, 辺数: {G.number_of_edges()}")

    df_cent = compute_centralities(G)

    # ハブらしさが分かりやすいように、重み付き in-degree 降順で並べておく
    df_cent = df_cent.sort_values("weighted_in_degree", ascending=False)

    # 出力ファイル名の決定
    base = os.path.basename(input_name)
    if base.startswith("similar_"):
        suffix = base[len("similar_"):]  # 例: "resnet50.csv"
        output_name = "centrality_" + suffix
    else:
        output_name = "centrality_" + base

    output_csv = os.path.join(script_dir, output_name)
    df_cent.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 中心性CSVを出力しました: {output_csv}")

    # 上位10件を表示
    print("\n[TOP 10 by weighted_in_degree]")
    print(df_cent.head(10).to_string(index=False))
    print()


def main():
    # スクリプトと同じディレクトリを基準にする
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 同じディレクトリにある複数の similar_*.csv に適用する
    input_files = [
        "rank_weighted_neighbors_from_keyphrases.csv",
    ]

    for name in input_files:
        process_one_file(script_dir, name)


if __name__ == "__main__":
    main()
