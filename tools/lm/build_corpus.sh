#!/bin/bash
# Build the Traditional Chinese (Taiwan) sentence corpus from zh Wikipedia.
#   1. prepare.py extracts sentences from the Wikimedia parquet dump
#   2. sentences already in Traditional script get OpenCC t2tw (char variants only)
#   3. Simplified sentences get OpenCC s2twp (chars + Taiwan phrases)
# Splitting by script avoids s2twp mangling already-Traditional phrases (e.g. 導出→匯出).
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-data/sentences_tw.txt}
LIMIT=${LIMIT:-0}
.venv/bin/python prepare.py --output data/raw.txt --limit-articles "$LIMIT"
opencc -c s2t.json -i data/raw.txt -o data/raw_s2t.txt 2>/dev/null
paste -d '\t' data/raw.txt data/raw_s2t.txt | LC_ALL=C awk -F '\t' '$1==$2 {print $1 > "data/trad.txt"; next} {print $1 > "data/simp.txt"}'
opencc -c t2tw.json  -i data/trad.txt -o data/trad_tw.txt 2>/dev/null
opencc -c s2twp.json -i data/simp.txt -o data/simp_tw.txt 2>/dev/null
cat data/trad_tw.txt data/simp_tw.txt | awk 'NF' > "$OUT"
echo "trad=$(wc -l < data/trad.txt) simp=$(wc -l < data/simp.txt) total=$(wc -l < "$OUT")"
rm -f data/raw.txt data/raw_s2t.txt data/trad.txt data/simp.txt data/trad_tw.txt data/simp_tw.txt
