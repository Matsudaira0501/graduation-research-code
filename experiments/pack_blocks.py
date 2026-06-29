import csv
import argparse


def read_blocks(path):
    
    """
    lv1_output.txt のようなファイルを読み込んで、
    「# の前まで」を 1ブロックとしてリストで返す。
    """
    # 文字コードは UTF-8 を試してダメなら CP932
    for enc in ("utf-8", "cp932"):
        try:
            with open(path, encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise

    blocks = []
    buf = []

    for line in lines:
        if line.strip() == "#":
            # 区切りに到達したら、直前までを 1ブロックとして確定
            block_text = "".join(buf).strip()
            if block_text:
                blocks.append(block_text)
            buf = []
        else:
            buf.append(line)

    # 最後に # がない場合の取りこぼし対策
    block_text = "".join(buf).strip()
    if block_text:
        blocks.append(block_text)

    return blocks


def write_blocks_to_csv(blocks, output_path):
    """
    各ブロックを 1セルにした CSV を出力する。
    行: ブロック番号
    列A: 1セルにまとまったテキスト（セル内改行付き）
    """
    with open(output_path, "w", newline="", encoding="cp932") as f:
        writer = csv.writer(f)
        for block in blocks:
            # block の中には '\n' が含まれているので、
            # Excel では「セル内改行付きの1セル」として扱われる
            writer.writerow([block])


def main(input_path, output_path):
    blocks = read_blocks(input_path)
    write_blocks_to_csv(blocks, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="# 区切りのキーフレーズ出力を、1ブロック=1セルのCSVに変換するスクリプト"
    )
    parser.add_argument("input", help="入力テキストファイル（lv1_output.txt など）")
    parser.add_argument("output", help="出力 CSV ファイル名（例: lv1_blocks.csv）")
    args = parser.parse_args()

    main(args.input, args.output)
