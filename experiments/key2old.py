from keyphrase_extractors import (
    EmbeddingModel,
    EmbeddingPrompts,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import SentenceEmbeddingBasedExtractionConfig

input_text = "左の図は「元号」を表していました。左から順に、「めいじ」「たいしょう」「しょうわ」「へいせい」「れいわ」と埋まります。よって、青矢印が通る文字を当てはめて、答えは「西暦（せいれき）」となります。"


embedding_model_config = EmbeddingModel(
    name="cl-nagoya/ruri-base",
    device="cpu",
    prompts=EmbeddingPrompts(query="クエリ: ", passage="文章: "),
    trust_remote_code=True,
    batchsize=32,
    show_progress_bar=False,
)
extraction_config = SentenceEmbeddingBasedExtractionConfig(
    diversity_mode="normal",
    max_filtered_phrases=30,
    max_filtered_sentences=30,
    threshold=0.7,
    filter_sentences=True,
    grammar_phrasing=True,
    ngram_range=None,
    use_masked_distance=False,
    minimum_characters=100,
)
extractor = SentenceEmbeddingBasedExtractor(
    model_config=embedding_model_config,
    extraction_config=extraction_config,
    max_characters=10000,
    stop_words=None,
    flat_output=True,
    use_order=False,
    logger=None,
)

keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=10)
print("cl-nagoya/ruri-base")

results = keyphrases.keyphrases
if results and isinstance(results[0], list):
    results = results[0]

for rank, kp in enumerate(results, start=1):
    # score を小数3桁に丸めて表示
    print(f"{rank:2d}. {kp.phrase}\t(score={kp.score:.3f})")

print("-" * 80)