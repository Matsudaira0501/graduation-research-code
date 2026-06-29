from keyphrase_extractors import ClassicalExtractor
from pke.unsupervised import TextRank

input_text = "マス目は「九九表」を表していました。右上は「1×9（いんく）」のマスのため、「インク」、その下は「2×9（にく）」のマスのため「肉」が書かれています。その他のイラストも「4×4（しし）」「9×2（くに）」「8×8（はっぱ）」を表しています。よって、左上のハテナは「3×5」のマスから伸びているため、答えは「サンゴ」です。"

extractor = ClassicalExtractor(
    extractor=TextRank(),
    stop_words=None,
    args_candidate_selection={"pos": {"NOUN", "PROPN", "ADJ", "NUM"}},
    args_candidate_weighting={
        "window": 2,
        "pos": {"NOUN", "PROPN", "ADJ", "NUM"},
        "top_percent": None,
        "normalized": False,
    },
    flat_output=True,
    use_order=False,
    logger=None,
)
keyphrases = extractor.get_keyphrase(input_text=input_text, top_n_phrases=30)
print("TextRank")
print(keyphrases.keyphrases)
print("-" * 80)