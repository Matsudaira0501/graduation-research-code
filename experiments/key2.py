import argparse

from keyphrase_extractors import (
    EmbeddingModel,
    EmbeddingPrompts,
    SentenceEmbeddingBasedExtractor,
)
from keyphrase_extractors.embedding_based import (
    SentenceEmbeddingBasedExtractionConfig,
)


def build_extractor():
    """SentenceEmbeddingBasedExtractor を作るヘルパー。"""
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
        minimum_characters=100,  # 100文字未満の文は抽出しない設定に注意
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
    return extractor


def read_texts_from_file(path):
    """
    入力ファイルから複数文書を読み込む。

    - 1行1文書
    - もしくは " で囲まれた複数行を 1 文書

    の両方をある程度扱える。
    文字コードは UTF-8 を試してダメなら CP932 を使う。
    """
    texts = []
    buf = []
    in_quote_block = False

    # 文字コード自動切り替え
    for enc in ("utf-8", "cp932"):
        try:
            with open(path, encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # ブロック外での空行はスキップ
        if not line.strip() and not in_quote_block:
            continue

        if not in_quote_block:
            stripped = line.strip()
            if stripped.startswith('"') and not stripped.endswith('"'):
                # " で始まり終わっていない → ブロック開始
                in_quote_block = True
                buf.append(stripped.lstrip('"'))
            elif stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 1:
                # 1行で "..." が完結している
                texts.append(stripped.strip('"'))
            else:
                # 普通の1行1文書
                texts.append(stripped)
        else:
            # ブロック内
            stripped = line
            if stripped.endswith('"'):
                buf.append(stripped.rstrip('"'))
                doc = "\n".join(buf).strip()
                if doc:
                    texts.append(doc)
                buf = []
                in_quote_block = False
            else:
                buf.append(stripped)

    # 閉じ忘れ対策
    if buf:
        doc = "\n".join(buf).strip()
        if doc:
            texts.append(doc)

    return texts


def write_keyphrases_for_texts(texts, output_path, extractor, top_n=10):
    """
    texts の各文書についてキーフレーズを抽出し、
    1件ずつ

      1. フレーズ (score=0.123)

    という形式で output_path に書き出す。
    空のフレーズは出力しない。
    """
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        for doc_id, text in enumerate(texts, start=1):
            # 空の文書はスキップ
            if not text.strip():
                out.write("\n")
                continue

            # 空フレーズを除外しても top_n 件残るように、少し多めに取得
            raw_top_n = top_n * 3

            keyphrases = extractor.get_keyphrase(
                input_text=text,
                top_n_phrases=raw_top_n,
            )

            results = keyphrases.keyphrases
            if results and isinstance(results[0], list):
                results = results[0]

            # 空文字・空白だけのフレーズを除外
            valid_results = [
                kp for kp in results
                if kp.phrase and kp.phrase.strip()
            ]

            # 上位 top_n 件だけ出力
            for rank, kp in enumerate(valid_results[:top_n], start=1):
                out.write(f"{rank:2d}. {kp.phrase}\t(score={kp.score:.3f})\n")

            out.write("#\n")  # 文書間の空行


def main(input_path, output_path, top_n=10):
    texts = read_texts_from_file(input_path)
    extractor = build_extractor()
    write_keyphrases_for_texts(texts, output_path, extractor, top_n=top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="埋め込みモデル(cl-nagoya/ruri-base)でキーフレーズ抽出してテキスト出力するスクリプト"
    )
    parser.add_argument("input", help="入力テキストファイル（lv1.txt など）")
    parser.add_argument("output", help="出力テキストファイル名")
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="各文書あたりのキーフレーズ上位件数（デフォルト: 10）",
    )
    args = parser.parse_args()

    main(args.input, args.output, top_n=args.top_n)
