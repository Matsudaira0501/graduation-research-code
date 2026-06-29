import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


def build_model(device: str = "cpu"):
    """
    ImageNet 学習済み ResNet50 を特徴抽出器として用意する。
    出力は 2048 次元ベクトル（最後の全結合層を Identity に差し替え）。
    """
    # torchvision のバージョンによって API が微妙に違うので、両対応気味に書いておく
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except AttributeError:
        # 古いバージョン用のフォールバック
        model = models.resnet50(pretrained=True)

    # 最後の分類層を Identity にして特徴ベクトルだけ取り出す
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def build_transform():
    """
    ResNet 用の前処理（ImageNet の標準前処理）を定義。
    """
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform


@torch.no_grad()
def extract_feature(path: str, model, transform, device: str = "cpu") -> np.ndarray:
    """
    1 枚の画像から ResNet 特徴 (2048 次元) を抽出して L2 正規化して返す。
    """
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    feat = model(x).cpu().numpy().astype("float32")[0]  # (2048,)
    # L2 正規化
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat


def collect_image_paths(img_dir: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(img_dir, ext)))
    paths = sorted(paths)
    if len(paths) == 0:
        raise ValueError(f"画像が見つかりませんでした: {img_dir}")
    return paths


def compute_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    features: shape (N, D), 各行は L2 正規化済み特徴ベクトル
    コサイン類似度行列 (N, N) を返す。
    """
    # L2 正規化済みなら内積 = コサイン類似度
    sim = features @ features.T  # (N, N)
    return sim


def build_topk_dataframe(img_paths, sim_matrix, top_k: int = 5) -> pd.DataFrame:
    """
    各画像について、類似度上位 top_k 件（自分自身を除く）をまとめた DataFrame を作る。
    """
    n = len(img_paths)
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]

    rows = []
    for i in range(n):
        sims = sim_matrix[i].copy()
        # 自分自身の類似度を -inf にしてランキングから除外
        sims[i] = -1.0
        # 類似度の高い順にインデックスを並べる
        idx_sorted = np.argsort(-sims)  # 降順
        top_indices = idx_sorted[:top_k]
        top_sims = sims[top_indices]

        row = {
            "query": base_names[i],
        }
        for rank, (j, s) in enumerate(zip(top_indices, top_sims), start=1):
            row[f"neighbor{rank}"] = base_names[j]
            row[f"sim{rank}"] = float(s)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="ResNet50 を用いた画像類似度検索（見た目ベース）"
    )
    parser.add_argument("--img_dir", type=str, required=True,
                        help="入力画像を格納したディレクトリ（61枚など）")
    parser.add_argument("--output_csv", type=str, default="similar_topk_resnet.csv",
                        help="類似画像リストを書き出す CSV ファイル名")
    parser.add_argument("--top_k", type=int, default=5,
                        help="各画像について何件の類似画像を出力するか（自分自身を除く）")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu / cuda。未指定なら自動判定")

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    img_paths = collect_image_paths(args.img_dir)
    print(f"画像枚数: {len(img_paths)}")

    model = build_model(device)
    transform = build_transform()

    # 全画像の特徴抽出
    features = []
    for path in img_paths:
        feat = extract_feature(path, model, transform, device)
        features.append(feat)
    features = np.stack(features)  # (N, D)

    # 類似度行列を計算
    sim_matrix = compute_similarity_matrix(features)

    # 各画像の top-k 類似画像を DataFrame にまとめる
    df_topk = build_topk_dataframe(img_paths, sim_matrix, top_k=args.top_k)

    # CSV に出力
    df_topk.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"CSV を出力しました: {args.output_csv}")


if __name__ == "__main__":
    main()
