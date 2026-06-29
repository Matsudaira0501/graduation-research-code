# Graduation Research Code

卒業研究で使用した画像類似度計算、画像検索、説明文処理、謎解き評価のコードです。

## Structure

- `src/`: 主な実行コード
- `experiments/`: 実験・評価・補助スクリプト
- `tools/`: ResNet画像類似度計算などの独立ツール

## Setup

Python 3.10以降を推奨します。

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

OpenAI APIを利用する処理では、キーをコードに直接書かず、`OPENAI_API_KEY` 環境変数を設定してください。

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Data

研究データ、画像、生成結果、学習済みモデルは、サイズ・著作権・プライバシーの都合により含めていません。

## Notes

- 依存関係はimport文から整理しているため、一部の実験では追加パッケージが必要になる場合があります。
- `keyphrase_extractors` と `pke` を使うコードは、対応パッケージを別途導入してください。
- APIキーなどの秘密情報をコミットしないでください。
