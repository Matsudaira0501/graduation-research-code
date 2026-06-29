# build_index_qwenvl.py
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def l2_normalize_mat(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: (B, L, H)
    attention_mask:   (B, L)
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
    denom = mask.sum(dim=1).clamp(min=1.0)                           # (B, 1)
    return summed / denom


def resize_long_side(image: Image.Image, max_side: int) -> Image.Image:
    """長辺が max_side を超える場合のみ縮小する。"""
    if max_side <= 0:
        return image
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / m
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h))


def _get_last_hidden_from_model(model: Qwen3VLForConditionalGeneration, inputs: dict) -> torch.Tensor:
    """
    Qwen3VLForConditionalGeneration から最終層 hidden state (B, L, H) を取り出す。
    transformers の差異に備えて複数パスを用意。
    """
    # 1) 多くのHF生成モデルは model.model が本体
    if hasattr(model, "model"):
        out = model.model(**inputs, return_dict=True)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state

    # 2) 直接 forward で hidden_states を要求
    out = model(**inputs, return_dict=True, output_hidden_states=True)
    if hasattr(out, "hidden_states") and out.hidden_states is not None and len(out.hidden_states) > 0:
        return out.hidden_states[-1]

    raise RuntimeError("Could not obtain last hidden state from the model outputs.")


@torch.inference_mode()
def embed_image_text(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image | None,
    text: str,
    device: torch.device,
    pooling: str = "mean",  # "mean" or "last"
) -> np.ndarray:
    """
    画像＋テキスト（解説文そのまま）を1本のベクトルへ。
    長文対策なし（truncationは保険で有効化）。
    """
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if image is None:
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            truncation=True,
        )
    else:
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
            truncation=True,
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    last = _get_last_hidden_from_model(model, inputs)  # (1, L, H)
    attn = inputs.get("attention_mask", None)

    if pooling == "mean" and attn is not None:
        pooled = mean_pool(last, attn)  # (1, H)
    else:
        # last real token
        if attn is None:
            pooled = last[:, -1, :]
        else:
            idx = attn.sum(dim=1) - 1
            pooled = last[torch.arange(last.size(0), device=last.device), idx, :]

    v = pooled[0].float().cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def load_explanations(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 0].astype(str).tolist()


def resolve_image_path(image_dir: Path, idx1: int, pattern: str) -> Path:
    return image_dir / pattern.format(id=idx1)


def parse_dtype(dtype_str: str) -> torch.dtype | str:
    """
    dtype 引数として渡す値を決める。
    transformers 側が "auto" を受け取れる場合があるので、auto は文字列のまま返す。
    """
    s = dtype_str.lower().strip()
    if s == "auto":
        return "auto"
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str} (use auto|fp16|bf16)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--image_pattern", type=str, default="{id:03d}.png")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out_dir", type=str, default="index_out_qwenvl")
    ap.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--dtype", type=str, default="fp16", help="auto / fp16 / bf16  (3070 8GBなら fp16 推奨)")
    ap.add_argument("--max_side", type=int, default=1024, help="画像の長辺をこの値に制限（<=0で無効）")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = load_explanations(csv_path)
    n = len(texts)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = parse_dtype(args.dtype)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, dtype=dtype)
    model.to(device)
    model.eval()

    print(f"[INFO] device={device}  cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")
    try:
        print(f"[INFO] dtype={next(model.parameters()).dtype}")
    except Exception:
        pass

    vecs = []
    metas = []

    t0 = time.time()

    for i in range(n):
        idx1 = i + 1
        img_path = resolve_image_path(image_dir, idx1, args.image_pattern)
        if not img_path.exists():
            raise FileNotFoundError(f"画像が見つかりません: {img_path}（CSV {idx1}行目に対応）")

        # ファイルハンドルを確実に閉じる
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        # OOM対策：必要なら縮小
        image = resize_long_side(image, args.max_side)

        v = embed_image_text(model, processor, image=image, text=texts[i], device=device, pooling=args.pooling)
        vecs.append(v)

        metas.append({
            "id": idx1,
            "image_path": str(img_path.as_posix()),
            "text": texts[i],
        })

        if idx1 % 10 == 0 or idx1 == n:
            elapsed = time.time() - t0
            per_item = elapsed / idx1
            remaining = per_item * (n - idx1)
            print(f"[INFO] embedded {idx1}/{n}  elapsed={elapsed:.1f}s  per_item={per_item:.2f}s  ETA={remaining/60:.1f}min")

    mat = np.vstack(vecs).astype(np.float32)
    mat = l2_normalize_mat(mat)

    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)  # cos類似度（正規化済み内積）
    index.add(mat)

    faiss.write_index(index, str((out_dir / "faiss.index").as_posix()))
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "csv": str(csv_path.as_posix()),
                "image_dir": str(image_dir.as_posix()),
                "image_pattern": args.image_pattern,
                "n": n,
                "dim": dim,
                "pooling": args.pooling,
                "device": str(device),
                "dtype": args.dtype,
                "max_side": args.max_side,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[OK] index created")
    print(f" out_dir={out_dir}  n={n}  dim={dim}  pooling={args.pooling}")


if __name__ == "__main__":
    main()
