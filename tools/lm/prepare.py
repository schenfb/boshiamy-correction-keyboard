#!/usr/bin/env python3
"""Extract Traditional-Chinese-friendly sentences from the Wikimedia zh Wikipedia parquet dump.

Source: https://huggingface.co/datasets/wikimedia/wikipedia (config 20231101.zh)
License of source text: CC BY-SA 4.0 (Wikipedia). Output is sentence-per-line text
that is then converted with OpenCC `s2twp` (see Makefile / README).
"""
import argparse
import glob
import re
import sys

import pyarrow.parquet as pq

SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
# Drop bracketed asides (mostly pinyin / dates / foreign names) which are noise for a keyboard LM.
PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
WS_RE = re.compile(r"\s+")


def sentences(text: str, min_len: int, max_len: int, min_cjk: float):
    for raw in SPLIT_RE.split(text):
        s = PAREN_RE.sub("", raw)
        s = WS_RE.sub("", s).strip("，、：,:「」『』\"'“”‘’")
        n = len(s)
        if n < min_len or n > max_len:
            continue
        cjk = len(CJK_RE.findall(s))
        if cjk / n < min_cjk:
            continue
        yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="data/hf/20231101.zh/*.parquet")
    ap.add_argument("--output", default="data/sentences_raw.txt")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--min-cjk", type=float, default=0.8)
    ap.add_argument("--limit-articles", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        sys.exit(f"no parquet files match {args.input_glob}")

    n_art = n_sent = n_chars = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    n_art += 1
                    for s in sentences(text, args.min_len, args.max_len, args.min_cjk):
                        out.write(s)
                        out.write("\n")
                        n_sent += 1
                        n_chars += len(s)
                    if args.limit_articles and n_art >= args.limit_articles:
                        break
                if args.limit_articles and n_art >= args.limit_articles:
                    break
            print(f"{f}: articles={n_art} sentences={n_sent} chars={n_chars}", file=sys.stderr)
            if args.limit_articles and n_art >= args.limit_articles:
                break
    print(f"done articles={n_art} sentences={n_sent} chars={n_chars}", file=sys.stderr)


if __name__ == "__main__":
    main()
