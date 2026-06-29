import csv
from openai import OpenAI

client = OpenAI(
    api_key="",
)

# ===== 1. lv1.txt を読み込む =====

def read_text_file_auto_encoding(path: str) -> str:
    """
    日本語テキストファイルをそれっぽいエンコーディングで読み込む。
    （Windows を想定して cp932 → utf-8 → shift_jis の順に試す）
    """
    encodings_to_try = ["cp932", "utf-8", "shift_jis"]
    last_error = None

    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            print(f"[INFO] encoding='{enc}' で読み込み成功")
            return text
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error or RuntimeError("どのエンコーディングでも読み込めませんでした。")


# ===== 2. lv1.txt 内の解説文を分割 =====

def parse_explanations(raw_text: str):
    """
    lv1.txt 内のテキストから「解説文のリスト」を取り出す。

    仕様:
      - 行頭が " で始まる行 → 引用ブロック開始
      - 引用ブロックは最後が " で終わる行 or 単独の " 行で閉じる
      - 上記に当てはまらない非空行は、それだけで 1 つの解説とみなす
    """
    lines = raw_text.splitlines()
    explanations = []
    buf = []
    in_quote_block = False

    for line in lines:
        if not in_quote_block:
            # まだ引用ブロック外
            if line.startswith('"'):
                in_quote_block = True
                stripped = line.strip()

                # 行全体が " だけ → 中身なし
                if stripped == '"':
                    continue

                # 1行の中で "〜" が完結しているパターン
                if stripped.endswith('"') and len(stripped) > 1:
                    content = stripped
                    if content.startswith('"'):
                        content = content[1:]
                    if content.endswith('"'):
                        content = content[:-1]
                    explanations.append(content.strip())
                    in_quote_block = False
                    buf = []
                else:
                    # 複数行ブロックの最初の行
                    first_quote_idx = line.find('"')
                    content = line[first_quote_idx + 1 :]
                    buf.append(content)
            else:
                # 引用なしの単独行 → そのまま 1 解説
                if line.strip():
                    explanations.append(line.strip())
        else:
            # 引用ブロック内
            stripped = line.strip()

            # 単独の " 行 → ブロック終了
            if stripped == '"':
                explanations.append("\n".join(buf).strip())
                buf = []
                in_quote_block = False

            # 行末が " で終わる → 同じ行でブロック終了
            elif stripped.endswith('"'):
                last_quote_idx = line.rfind('"')
                content = line[:last_quote_idx]
                buf.append(content)
                explanations.append("\n".join(buf).strip())
                buf = []
                in_quote_block = False

            else:
                # ブロック継続
                buf.append(line)

    return explanations


# ===== 3. OpenAI API でキーフレーズ抽出 =====

def extract_keyphrases_from_explanation(
    explanation_text: str,
    top_k: int = 20,
    model: str = "gpt-5.1",   # ★ 実際に動いたモデルに戻す
):
    """
    1つの謎解き解説文から、top_k 個のキーフレーズを抽出する。

    戻り値: ["キーフレーズ1", "キーフレーズ2", ...]
    """

    prompt = f"""
あなたは日本語テキストからキーフレーズを抽出する専門家です。
次の「謎解きの解説文」から、内容理解に重要なキーフレーズを抽出してください。

【抽出方針】

最優先：謎解きの「仕掛け・ルール・発想の転換」を表す語句

次に優先：そのルールを示す代表的な「対応関係のペア／まとまり」

補助的に：最終的な答えや、解答に直接つながる重要な要素を含める

【まとめ方のルール】

個々の要素をバラバラに列挙するよりも、
「関係」や「対応」を表すフレーズ（例：数字と季節の対応）の方を優先する。

汎用的すぎる語（問題、解説、図、答え など）や、
この問題に特有でない表現は、{top_k} を埋めるための補助としてのみ使う。

【要件】
- 関連度が低くてもよいので補って必ず {top_k} 個にする。
- 「必ず」1〜5文字程度の「短い」名詞句・フレーズにする。
- 日本語のまま抽出する（翻訳しない）
- 類似した表現は代表的なもの1つにまとめる
- 謎解きで重要な「仕掛け・ルール・発想の転換」に関する語を優先する
- 重要度が高い順に並べる
- 出力は「各キーフレーズを1行に1つだけ書いたプレーンテキスト」とする
  - 行頭に番号やハイフン・中黒などの記号をつけない
  - 余計な説明文やコメントを書かない
  - 例：
    キーフレーズA
    キーフレーズB
    キーフレーズC

【謎解きの解説文】
\"\"\"{explanation_text}\"\"\""""

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=512,
    )

    # model 出力を確実にテキストとして取り出す
    raw_output = ""
    if response.output:
        for out in response.output:
            if getattr(out, "type", "") == "message":
                for content in (out.content or []):
                    if getattr(content, "type", "") == "output_text":
                        raw_output += content.text

    # デバッグ用に生出力を表示
    print("\n===== モデル生出力 =====")
    print(raw_output)
    print("===== モデル生出力ここまで =====\n")

    # 各行をキーフレーズとして解釈
    lines = [line.strip() for line in raw_output.splitlines()]
    keyphrases = [line for line in lines if line]  # 空行除去

    # top_k を超えていれば切る
    if len(keyphrases) > top_k:
        keyphrases = keyphrases[:top_k]

    return keyphrases


# ===== 4. メイン処理：2つだけ逆順 id で CSV に出力 =====

def main():
    input_path = "lv1.txt"                     # 入力テキストファイル
    output_path = "lv1_keyphrases_test2.csv"   # 出力CSV（テスト用）
    top_k = 20

    # 1) ファイル読み込み
    raw_text = read_text_file_auto_encoding(input_path)

    # 2) 解説リストに分割
    explanations = parse_explanations(raw_text)
    total = len(explanations)
    print(f"[INFO] 解説の数: {total}")

    if total == 0:
        print("[ERROR] 解説が1つも抽出できませんでした。")
        return

    # === テストなので「最初の2つの解説」だけ処理 ===
    target_explanations = explanations[:2]

    rows = []

    for local_idx, explanation in enumerate(target_explanations, start=1):
        # id を「61→1の逆順」にしたいので、
        # 全体数 total に対して「後ろから数えた番号」を付与
        # 例: total=61 のとき local_idx=1 → id=61, local_idx=2 → id=60
        output_id = total - local_idx + 1

        print(f"\n===== 解説 local_idx={local_idx}, 出力id={output_id} を処理中 =====")
        print(explanation)
        print("===================================================")

        keyphrases = extract_keyphrases_from_explanation(
            explanation,
            top_k=top_k,
        )

        print("抽出されたキーフレーズ:")
        for i, kp in enumerate(keyphrases, start=1):
            print(f"{i:2d}. {kp}")

        # 「1セル内にすべてまとめる」ための文字列を作成
        # 例:
        #   1. 九九表
        #   2. マス目
        block_lines = [
            f"{i}. {kp}" for i, kp in enumerate(keyphrases, start=1)
        ]
        block_text = "\n".join(block_lines)

        rows.append([output_id, block_text])

    # 念のため id の降順にソート（61, 60 の順になる想定）
    rows.sort(key=lambda r: r[0], reverse=True)

    # 4) CSV に保存（Excel 日本語環境を意識して cp932）
    header = ["id", "keyphrases"]

    with open(output_path, "w", newline="", encoding="cp932") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[INFO] テスト用 CSV を書き出しました: {output_path}")


if __name__ == "__main__":
    main()
