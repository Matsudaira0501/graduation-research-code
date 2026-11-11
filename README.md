# 画像謎解きの解答支援 | 類題検索 × In-Context Learning

**画像の謎解き問題に対して、解法が似ている類題とその解説をLLMへ提示し、解答精度を改善する**ことを目指した卒業研究の実験コードです。

[研究発表の情報（MVE研究会）](https://ken.ieice.org/ken/index/ieice-techrep-125-411.html) ｜ [研究業績（名古屋大学）](https://www.cs.is.i.nagoya-u.ac.jp/ja/people/hayato-matsudaira/) ｜ [類題提示による解答プログラム](src/nazotoki-fewshot.py)

## 研究のポイント

- **課題：** 見た目が似た画像でも、謎の解き方が同じとは限らない
- **着想：** 対象問題とともに、検索した類題の**画像・解説・正解**をLLMに提示する
- **工夫：** タグ／キーフレーズ、画像特徴量、マルチモーダル埋め込みなど、異なる基準で類題を検索して比較する
- **評価：** 61問の画像謎解き問題で、類題提示の有無と検索方法ごとの差を調べる

### 実験結果（卒業研究時）

| 条件・指標 | 結果 |
| --- | --- |
| データセット | 画像謎解き61問 |
| 類題を提示しない基準条件 | **33/61問正解（約54%）** |
| 類題提示による改善 | **最大約15ポイント** |

結果は研究時の評価によるもので、**すべての検索方法で改善したことを意味しません**。また、研究データ・当時の全実験設定・生成結果一式は公開していないため、このREADMEのみで数値を再現できる状態ではありません。

## 処理の流れ

```text
対象の画像謎解き問題
         │
         ├── 類題検索 ─────────────────────────┐
         │   ・解説のキーフレーズ／タグ          │
         │   ・画像特徴（ResNet50、DINOv2等）    │
         │   ・画像・言語の埋め込み（Qwen-VL系）  │
         │                                     ▼
         └──────────────────→ 類題の画像・解説・答え
                                               │
                                               ▼
                               対象画像と一緒にLLMへ入力
                                               │
                                               ▼
                                     生成された解答
                                               │
                                               ▼
                                    正解データとの比較
```

画像の類似性だけでなく、**解法の類似性をどう検索結果へ反映させるか**を検討した点が本研究の中心です。

## 主な実装

| コード | 役割 |
| --- | --- |
| [src/nazotoki.py](src/nazotoki.py) | 類題を提示しない基準実験 |
| [src/nazotoki-fewshot.py](src/nazotoki-fewshot.py) | 類題の画像・解説・答えを提示する解答実験 |
| [src/image_similar.py](src/image_similar.py) | ResNet50、ViT、DINOv2による特徴抽出・画像類似度 |
| [src/explanation_keyphrases.py](src/explanation_keyphrases.py) | 解説文からのキーフレーズ抽出 |
| [src/build_index_qwenvl.py](src/build_index_qwenvl.py) / [src/search_qwenvl.py](src/search_qwenvl.py) | 埋め込みによる索引作成・検索 |
| [src/seigohantei.py](src/seigohantei.py) | 正誤評価に関する処理 |
| [experiments/](experiments/) | 比較・評価用の実験スクリプト |
| [tools/](tools/) | 独立した補助ツール |

研究過程の複数の実験スクリプトを整理したコード集であり、単一コマンドで全処理を再現する構成ではありません。

## 使用技術

**Python / OpenAI API（研究時にGPT-5.1を使用） / PyTorch / torchvision / ResNet50 / DINOv2 / Qwen-VL系の埋め込み / FAISS / pandas / scikit-learn** など。

## セットアップと実行の例

Python 3.10以降を推奨します。

```bash
git clone https://github.com/Matsudaira0501/graduation-research-code.git
cd graduation-research-code
python -m pip install -r requirements.txt
```

OpenAI APIを使用する場合は、**APIキーをコードへ直接書かず**、環境変数 `OPENAI_API_KEY` を設定してください（[.env.example](.env.example) 参照）。

類題提示による実験の呼び出し例：

```bash
python src/nazotoki-fewshot.py \
  --dir /path/to/images \
  --similar /path/to/similar.csv \
  --answer /path/to/answer.csv \
  --kaisetu /path/to/kaisetu.csv \
  --model gpt-5.1 \
  --out results_fewshot.csv
```

- `--dir`：数値IDを含むファイル名の問題画像を置いたフォルダ
- `--similar`：`query` と `neighbor1` 列を持つ類題対応CSV
- `--answer` / `--kaisetu`：問題ID順の正解・解説CSV（それぞれヘッダーなし・1列）
- `--out`：解答結果の出力先CSV

**この実行例は必要な引数を示すものであり、入力データを別途用意する必要があります。** 一部スクリプトでは追加パッケージ（例：`pke`、`keyphrase_extractors`）、モデル重みや個別の設定変更も必要です。APIの利用料金にも注意してください。

## 学会・セミナーでの発表

- **電子情報通信学会 MVE研究会（2026年3月、口頭発表）**  
  「単一画像謎解きに対する類題検索とインコンテキスト学習を用いた解答支援」  
  技術研究報告 **MVE2025-56**（[掲載号の発表一覧](https://ken.ieice.org/ken/index/ieice-techrep-125-411.html)）
- **2nd Taiwan-Japan Multimedia Seminar（TJMM2026、英語ポスター発表）**  
  “Image Puzzle Solving by In-context Learning of Similar-puzzle” （[研究業績の掲載ページ](https://www.cs.is.i.nagoya-u.ac.jp/ja/people/hayato-matsudaira/)）

上記のリンクは発表実績の確認用です。発表スライド・ポスター・データセットを、このリポジトリ内で配布しているわけではありません。

## データ公開について

問題画像、解説・正解データ、生成結果、学習済みモデルなどは、著作権・プライバシー・容量の都合により含めていません。公開されているコードと研究実験の全データは区別してください。

---

関連作品：[Unity / C#：2Dアクションゲーム（ブラウザでプレイ可）](https://github.com/Matsudaira0501/2Dgame)
