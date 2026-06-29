import csv
from openai import OpenAI

# ★ここに自分の API キーを入れるか、環境変数 OPENAI_API_KEY を設定しておく
client = OpenAI()


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
    model: str = "gpt-5.1",
):
    """
    1つの謎解き解説文から、top_k 個のキーフレーズを抽出する。

    戻り値: ["キーフレーズ1", "キーフレーズ2", ...]
    """

    prompt = f"""
あなたは日本語テキストからキーフレーズを抽出する専門家です。
次の「謎解きの解説文」から、内容理解に重要なキーフレーズを抽出してください。

【抽出の目的】
このキーフレーズは「類似問題検索」に使われます。
同じタイプの謎からは、できるだけ「同じ表現」のキーフレーズが出るようにしてください。

【抽出方針】

最優先：謎解きの「仕掛け・ルール・発想の転換」を表す語句
次に優先：そのルールを示す代表的な「対応関係のペア／まとまり」
補助的に：最終的な答えや、解答に直接つながる重要な要素

【粒度・形のルール（重要）】

- 各キーフレーズは「1〜8文字程度」の短い名詞句・フレーズにする
  - 例：「五十音表」「デジタル数字」「漢字の部首」「左右の手」
- 「〜を使った問題」「〜を表している図」などの説明は付けず、核となる名詞だけにする
  - 悪い例：「五十音表を使った問題」→ 良い例：「五十音表」
- 1つのキーフレーズには、できるだけ「1つの概念だけ」を含める
  - 悪い例：「五十音表と九九表」→ 良い例：「五十音表」「九九表」（2つに分ける）
- 類似した表現は代表的なもの1つにまとめる
  - 例：「ひらがな表」「あいうえお表」→「五十音表」に一本化
- 「10文字以上になりそうな場合は、より短い言い換えにすること」

【優先順位】

- まず、その謎の仕掛けを特徴付ける「標準表現」をできるだけ挙げる
- 余裕があれば、対応関係や重要な要素（例：「1文字目」「指の本数」「棒の本数」など）を加える
- どうしても {top_k} 個に足りないときのみ、汎用的な語を補助的に使う

【出力形式】

- 関連度が低くてもよいので補って必ず {top_k} 個にする
- 「重要度が高い順に並べる」
- 出力は「各キーフレーズを1行に1つだけ書いたプレーンテキスト」とする
  - 行頭に番号や記号をつけない
  - 余計な説明文やコメントを書かない
  - 例：
    五十音表
    九九表
    デジタル数字
    左右の手
    指の本数

- 日本語のまま抽出する（翻訳しない）

【謎解きの解説文】
\"\"\"{explanation_text}\"\"\""""

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=5120,
        reasoning={"effort": "high"},
    )

    # model 出力を確実にテキストとして取り出す
    raw_output = ""
    if response.output:
        for out in response.output:
            if getattr(out, "type", "") == "message":
                for content in (out.content or []):
                    if getattr(content, "type", "") == "output_text":
                        raw_output += content.text

    # 必要に応じてコメントアウト
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


# ===== 4. メイン処理：全 61 件を id=61→1 の逆順で CSV に出力 =====

def main():
    #input_path = "lv1.txt"
    input_path = "text2.txt"                         # 入力テキストファイル
    output_path = "2.csv"         # 出力CSV（本番）
    top_k = 10

    # 1) ファイル読み込み
    raw_text = read_text_file_auto_encoding(input_path)

    # 2) 解説リストに分割
    explanations = parse_explanations(raw_text)
    total = len(explanations)
    print(f"[INFO] 解説の数: {total}")

    if total == 0:
        print("[ERROR] 解説が1つも抽出できませんでした。")
        return

    rows = []

    # === 全 61 件を処理 ===
    for idx, explanation in enumerate(explanations, start=1):
        # id を「61→1の逆順」にする
        # 例: total=61 のとき idx=1 → id=61, idx=2 → id=60, ..., idx=61 → id=1
        output_id = total - idx + 1

        print(f"\n===== 解説 idx={idx}, 出力id={output_id} を処理中 =====")
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

    # 念のため id の降順でソート（61, 60, ..., 1）
    rows.sort(key=lambda r: r[0], reverse=True)

    # 4) CSV に保存（Excel 日本語環境を意識して cp932）
    header = ["id", "keyphrases"]

    with open(output_path, "w", newline="", encoding="cp932") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[INFO] 全 61 件のキーフレーズを CSV に書き出しました: {output_path}")


if __name__ == "__main__":
    main()
