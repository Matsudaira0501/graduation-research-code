import os
import argparse
from glob import glob
import shutil

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# ---------------------------------------------------------
#  モデル & 前処理の構築（backbone を切り替える）
# ---------------------------------------------------------
def build_model_and_transform(backbone: str, device: str = "cpu"):
    """
    backbone に応じて特徴抽出モデルと前処理 transform を返す。

    対応:
      - resnet50      : ImageNet pretrain
      - vit_b16       : ImageNet pretrain ViT-B/16
      - dinov2_vits14 : DINOv2 ViT-S/14 (torch.hub 経由)
    """
    backbone = backbone.lower()

    if backbone == "resnet50":
        # ResNet50
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
            transform = weights.transforms()
        except AttributeError:
            # 古い torchvision の場合
            model = models.resnet50(pretrained=True)
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        # 最後の全結合層を Identity にして特徴ベクトルだけを出す
        model.fc = nn.Identity()

    elif backbone == "vit_b16":
        # Vision Transformer (ViT-B/16)
        try:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights)
            transform = weights.transforms()
        except AttributeError:
            # 古い torchvision 用フォールバック
            model = models.vit_b_16(pretrained=True)
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # 分類ヘッドを Identity に差し替え
        try:
            model.heads.head = nn.Identity()
        except AttributeError:
            model.heads = nn.Identity()

    elif backbone == "dinov2_vits14":
        # DINOv2 ViT-S/14 (facebookresearch/dinov2)
        # 初回実行時にインターネット経由で weight をダウンロードします
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # DINOv2 も 224x224 / ImageNet 正規化で問題ない
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        raise ValueError(f"未知の backbone 指定です: {backbone}")

    model.to(device)
    model.eval()
    return model, transform


# ---------------------------------------------------------
#  画像読み込み & 特徴抽出
# ---------------------------------------------------------
def collect_image_paths(img_dir: str):
    """
    img_dir 以下の画像をすべて集める。
    """
    print(f"[DEBUG] 検索ディレクトリ: {img_dir}")
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    paths = []
    for ext in exts:
        pattern = os.path.join(img_dir, ext)
        found = glob(pattern)
        print(f"[DEBUG] パターン {pattern} -> {len(found)} 件")
        paths.extend(found)

    paths = sorted(paths)
    if len(paths) == 0:
        raise ValueError(f"画像が見つかりませんでした: {img_dir}")
    print(f"[DEBUG] 見つかった画像枚数: {len(paths)}")
    return paths


@torch.no_grad()
def extract_feature(path: str, model, transform, device: str = "cpu") -> np.ndarray:
    """
    1 枚の画像から特徴ベクトルを抽出して L2 正規化して返す。
    """
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)
    feat = model(x).cpu().numpy().astype("float32")[0]
    # L2 正規化
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat


def compute_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    features: shape (N, D), 各行は L2 正規化済み特徴ベクトル
    -> コサイン類似度行列 (N, N)
    """
    return features @ features.T


def build_topk_dataframe(img_paths, sim_matrix, top_k: int = 5) -> pd.DataFrame:
    """
    各画像について、類似度上位 top_k 件（自分自身を除く）をまとめた DataFrame を作る。
    出力では拡張子を除いたファイル名（例: "1.webp" -> "1"）を使う。
    """
    n = len(img_paths)
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]

    rows = []
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # 自分自身を除外
        idx_sorted = np.argsort(-sims)  # 降順
        top_indices = idx_sorted[:top_k]
        top_sims = sims[top_indices]

        row = {"query": base_names[i]}
        for rank, (j, s) in enumerate(zip(top_indices, top_sims), start=1):
            row[f"neighbor{rank}"] = base_names[j]
            row[f"sim{rank}"] = float(s)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_similar_image_groups(
    img_paths,
    sim_matrix,
    output_root: str,
    top_k: int = 5,
    include_query: bool = True,
):
    """
    類似度行列に基づいて、各画像ごとに類似画像を集めたフォルダを作成する。
    ファイル名は順位付き (00_*, 01_*, ...) にする。
    """
    os.makedirs(output_root, exist_ok=True)

    n = len(img_paths)
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]

    for i in range(n):
        query_id = base_names[i]
        query_dir = os.path.join(output_root, query_id)
        os.makedirs(query_dir, exist_ok=True)

        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # 自分自身除外
        idx_sorted = np.argsort(-sims)
        top_indices = idx_sorted[:top_k]

        # (画像インデックス, ランク) のリストを作成
        raw_list = []

        if include_query:
            raw_list.append((i, 0))
            start_rank = 1
        else:
            start_rank = 0

        rank = start_rank
        for j in top_indices:
            raw_list.append((j, rank))
            rank += 1

        # 重複除去（同じ画像が複数回入るのを防ぐ）
        seen = set()
        id_rank_list = []
        for idx, r in raw_list:
            if idx in seen:
                continue
            seen.add(idx)
            id_rank_list.append((idx, r))

        # 実際にコピー
        for idx, r in id_rank_list:
            src = img_paths[idx]
            stem = base_names[idx]
            ext = os.path.splitext(src)[1]
            new_name = f"{r:02d}_{stem}{ext}"
            dst = os.path.join(query_dir, new_name)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"[INFO] コピー: {src} -> {dst}")

    print(f"[DONE] 類似画像フォルダを {output_root} 以下に作成しました。")


# ---------------------------------------------------------
#  メイン
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="任意の backbone (ResNet / ViT / DINOv2) を用いた画像類似度計算 + 類似画像コピー"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="入力画像を格納したディレクトリ（61枚など）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="類似画像リストを書き出す CSV ファイル名",
    )
    parser.add_argument(
        "--group_dir",
        type=str,
        required=True,
        help="似ている画像をコピーするルートディレクトリ（クエリごとのサブフォルダを作成）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="各画像について何件の類似画像を出力・コピーするか（自分自身を除く）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / cuda。未指定なら自動判定",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="特徴抽出バックボーン: resnet50 / vit_b16 / dinov2_vits14",
    )
    parser.add_argument(
        "--no_include_query",
        action="store_true",
        help="指定すると、クエリ画像自身はフォルダにコピーしない",
    )

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    include_query = not args.no_include_query

    print(f"[INFO] Device   : {device}")
    print(f"[INFO] Backbone : {args.backbone}")
    print(f"[INFO] Top-k    : {args.top_k}")
    print(f"[INFO] CSV      : {args.output_csv}")
    print(f"[INFO] GroupDir : {args.group_dir}")

    img_paths = collect_image_paths(args.img_dir)

    # モデル & 前処理を構築
    model, transform = build_model_and_transform(args.backbone, device)

    # 特徴抽出
    feats = []
    for p in img_paths:
        feat = extract_feature(p, model, transform, device)
        feats.append(feat)
    feats = np.stack(feats)  # (N, D)

    # 類似度行列
    sim_matrix = compute_similarity_matrix(feats)

    # top-k 類似画像 DataFrame
    df_topk = build_topk_dataframe(img_paths, sim_matrix, top_k=args.top_k)

    # CSV 書き出し
    df_topk.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] CSV を出力しました: {args.output_csv}")

    # 類似画像をフォルダにコピー
    save_similar_image_groups(
        img_paths,
        sim_matrix,
        output_root=args.group_dir,
        top_k=args.top_k,
        include_query=include_query,
    )


if __name__ == "__main__":
    main()
