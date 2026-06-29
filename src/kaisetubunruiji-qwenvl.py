import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 1) CSVを読む（ヘッダ無し）=====
df = pd.read_csv("kaisetu.csv", encoding="utf-8-sig", header=None)
texts = df[0].fillna("").astype(str).tolist()
ids = [str(i + 1) for i in range(len(texts))]

print("件数:", len(texts))  # ここが 61 になるのを確認

# ===== 2) Qwen3-Embeddingでベクトル化 =====
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
emb = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
emb = np.asarray(emb, dtype="float32")

# ===== 3) 類似度 =====
S = cosine_similarity(emb, emb)
np.fill_diagonal(S, -1.0)

# ===== 4) top1 =====
top1_idx = S.argmax(axis=1)
top1_score = S.max(axis=1)

out = pd.DataFrame({
    "id": ids,
    "most_similar_id": [ids[j] for j in top1_idx],
    "similarity": top1_score,
})
out.to_csv("most_similar_by_qwen8.csv", index=False, encoding="utf-8-sig")
print(out.head(10))
