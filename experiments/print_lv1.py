# check_lv1_explanations.py

from pathlib import Path

def read_text_file_auto_encoding(path: str) -> str:
    """
    日本語テキストファイルをそれっぽいエンコーディングで読み込む。
    （Windows なら cp932 が多いので、cp932 → utf-8 の順に試す）
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


def parse_explanations(raw_text: str):
    """
    lv1.txt 内のテキストから「解説のリスト」を取り出す。

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


def main():
    input_path = "lv1.txt"  # 必要ならフルパスに変更

    # 1) ファイル読み込み
    raw_text = read_text_file_auto_encoding(input_path)

    # 2) 解説に分割
    explanations = parse_explanations(raw_text)

    # 3) 個数と中身を確認用に表示
    print("===================================")
    print(f"解説の数: {len(explanations)}")
    print("===================================")

    for idx, exp in enumerate(explanations, start=1):
        print(f"\n---------- 解説 {idx} ----------")
        print(exp)
        print("------------------------------")

if __name__ == "__main__":
    main()
