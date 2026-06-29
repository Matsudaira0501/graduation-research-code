import argparse
import numpy as np
import torch
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

def read_text_auto(path: str) -> str:
    # Windowsでありがちな順に試す
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            pass
    # 最後の手段：文字化け回避で強制読み（置換）
    with open(path, "r", encoding="cp932", errors="replace") as f:
        return f.read()


def _get_last_hidden_from_model(model: Qwen3VLForConditionalGeneration, inputs: dict) -> torch.Tensor:
    if hasattr(model, "model"):
        out = model.model(**inputs, return_dict=True)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state

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
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
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


@torch.inference_mode()
def embed_image_only(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    device: torch.device,
    pooling: str = "mean",
) -> np.ndarray:
    # 画像だけでもテンプレを安定させるため最低1トークン
    content = [{"type": "image", "image": image}, {"type": "text", "text": " "}]
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    inputs = processor(text=prompt, images=[image], return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    last = _get_last_hidden_from_model(model, inputs)
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
    raise ValueError("dtype must be auto|fp16|bf16")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img14", type=str, default="14.webp")
    ap.add_argument("--img53", type=str, default="53.webp")
    ap.add_argument("--text14", type=str, default="text14.txt")
    ap.add_argument("--text53", type=str, default="text53.txt")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    ap.add_argument("--dtype", type=str, default="fp16", help="auto/fp16/bf16")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--max_side", type=int, default=1024)
    args = ap.parse_args()

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
    print(f"[INFO] dtype={next(model.parameters()).dtype}")

    text14 = read_text_auto(args.text14)
    text53 = read_text_auto(args.text53)
    
    img14 = Image.open(args.img14).convert("RGB")
    img53 = Image.open(args.img53).convert("RGB")
    img14 = resize_long_side(img14, args.max_side)
    img53 = resize_long_side(img53, args.max_side)

    # 1) image + text
    v14_it = embed_image_text(model, processor, img14, text14, device, pooling=args.pooling)
    v53_it = embed_image_text(model, processor, img53, text53, device, pooling=args.pooling)

    # 2) text only
    v14_t = embed_image_text(model, processor, None, text14, device, pooling=args.pooling)
    v53_t = embed_image_text(model, processor, None, text53, device, pooling=args.pooling)

    # 3) image only
    v14_i = embed_image_only(model, processor, img14, device, pooling=args.pooling)
    v53_i = embed_image_only(model, processor, img53, device, pooling=args.pooling)

    print("\n=== Cosine Similarities (14 vs 53) ===")
    print(f"cos(image+text): {cos(v14_it, v53_it):.6f}")
    print(f"cos(text only) : {cos(v14_t, v53_t):.6f}")
    print(f"cos(image only): {cos(v14_i, v53_i):.6f}")


if __name__ == "__main__":
    main()
