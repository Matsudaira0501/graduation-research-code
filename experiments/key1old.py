from keyphrase_extractors import ClassicalExtractor
from pke.unsupervised import TextRank

input_text = "それぞれの色の漢字を音読みで読む、という法則でした。「黒」と「白」は「コク」と「ハク」、「青」と「赤」は「セイ」と「セキ」になります。よって、「紫」と「緑」は「シ」と「リョク」になるので、答えは「シリョク」となります。"

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