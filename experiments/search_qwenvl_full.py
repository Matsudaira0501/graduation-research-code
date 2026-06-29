# batch_search_image_text_qwenvl.py
# 画像 + 解説文（CSVの1列目）をクエリとして、FAISS（同一方式で作ったインデックス）から近傍検索し、
# id, neighbor1..K, sim1..K をCSVに出力する。
#
# 前提:
# - build_index_qwenvl.py の出力ディレクトリ（faiss.index / meta.jsonl）がある
# - meta.jsonl には {"id":..., "image_path":..., "text":...} が入っている（あなたのインデックス作成コードの形式）
#
# 使い方例:
# python batch_search_image_text_qwenvl.py --index_dir index_out_qwenvl --out_csv image_text_results.csv --topk 10 --exclude_self --device cuda --dtype fp16 --max_side 1024

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import faiss
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
    denom = mask.sum(dim=1).clamp(min=1.0)                           # (B, 1)
    return summed / denom


def resize_long_side(image: Image.Image, max_side: int) -> Image.Image:
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
    pooling: str = "mean",
) -> np.ndarray:
    """
    画像 + テキストをインデックス作成時と同じ方式で1本のベクトルにする。
    """
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    # 空だとテンプレが不安定な場合があるので最低1トークン入れる
    content.append({"type": "text", "text": text if text else " "})

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if image is None:
        inputs = processor(text=prompt, return_tensors="pt", truncation=True)
    else:
        inputs = processor(text=prompt, images=[image], return_tensors="pt", truncation=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    last = _get_last_hidden_from_model(model, inputs)  # (1, L, H)
    attn = inputs.get("attention_mask", None)

    if pooling == "mean" and attn is not None:
        pooled = mean_pool(last, attn)
    else:
        if attn is None:
            pooled = last[:, -1, :]
        else:
            idx = attn.sum(dim=1) - 1
            pooled = last[torch.arange(last.size(0), device=last.device), idx, :]

    v = pooled[0].float().cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def parse_dtype(dtype_str: str) -> torch.dtype | str:
    s = dtype_str.lower().strip()
    if s == "auto":
        return "auto"
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str} (use auto|fp16|bf16)")


def load_meta(meta_path: Path) -> list[dict]:
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, required=True, help="build_index_qwenvl.py の out_dir")
    ap.add_argument("--out_csv", type=str, default="image_text_search_results.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--exclude_self", action="store_true", help="自己一致（同一id）を除外（推奨）")

    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    ap.add_argument("--dtype", type=str, default="fp16", help="auto / fp16 / bf16")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--max_side", type=int, default=1024, help="画像の長辺制限（<=0で無効）")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = parse_dtype(args.dtype)

    # load index + meta
    index = faiss.read_index(str(index_path.as_posix()))
    metas = load_meta(meta_path)

    # load model
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, dtype=dtype)
    model.to(device)
    model.eval()

    print(f"[INFO] device={device}  cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")
    print(f"[INFO] dtype={next(model.parameters()).dtype}")
    print(f"[INFO] queries={len(metas)}  topk={args.topk}  exclude_self={args.exclude_self}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["id"]
    for r in range(1, args.topk + 1):
        header.append(f"neighbor{r}")
    for r in range(1, args.topk + 1):
        header.append(f"sim{r}")

    raw_k = args.topk + 1 if args.exclude_self else args.topk

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for qpos, m in enumerate(metas):
            qid = m.get("id", qpos + 1)
            img_path = Path(m["image_path"]) if "image_path" in m else None
            qtext = m.get("text", "")

            if img_path is None or not img_path.exists():
                raise FileNotFoundError(f"query image not found for id={qid}: {img_path}")

            with Image.open(img_path) as im:
                qimg = im.convert("RGB")
            qimg = resize_long_side(qimg, args.max_side)

            qvec = embed_image_text(
                model=model,
                processor=processor,
                image=qimg,
                text=qtext,
                device=device,
                pooling=args.pooling,
            ).astype(np.float32)[None, :]

            scores, ids = index.search(qvec, raw_k)

            neighbors = []
            sims = []

            for hit_pos, score in zip(ids[0].tolist(), scores[0].tolist()):
                if hit_pos < 0 or hit_pos >= len(metas):
                    continue

                if args.exclude_self and hit_pos == qpos:
                    continue

                neighbors.append(metas[hit_pos].get("id", hit_pos + 1))
                sims.append(score)

                if len(neighbors) >= args.topk:
                    break

            while len(neighbors) < args.topk:
                neighbors.append("")
            while len(sims) < args.topk:
                sims.append("")

            writer.writerow([qid] + neighbors + sims)

            if (qpos + 1) % 10 == 0 or (qpos + 1) == len(metas):
                print(f"[INFO] processed {qpos+1}/{len(metas)}")

    print(f"[OK] saved: {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
