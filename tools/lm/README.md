# Language model pipeline

Builds the offline Traditional Chinese character trigram model (`.bslm`) used by
`boshiamy_core::NgramModel`. Everything runs locally; nothing here is shipped
with user data.

## Requirements

- `uv` (Python 3.12 venv), `opencc` CLI (`brew install opencc`), Rust toolchain
- ~2 GB disk for the Wikipedia parquet files, ~3 GB for intermediate text

## Steps

```bash
cd tools/lm
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python huggingface_hub pyarrow

# 1. Download zh Wikipedia (wikimedia/wikipedia 20231101.zh, CC BY-SA 4.0)
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('wikimedia/wikipedia', repo_type='dataset', allow_patterns=['20231101.zh/*'], local_dir='data/hf')"

# 2. Extract sentences and normalise to Taiwan Traditional Chinese
./build_corpus.sh data/sentences_tw.txt        # LIMIT=20000 for a quick sample

# 3. Train (from repo root)
cargo run --release -p boshiamy_lm_train -- \
  --input tools/lm/data/sentences_tw.txt --output tools/lm/data/zhwiki_tw.bslm \
  --min-bigram 3 --min-trigram 5 --max-bigrams 600000 --max-trigrams 1200000
```

The trainer prints held-out perplexity computed through the same loader the
app uses, so a format bug shows up as a perplexity blow-up.

## Evaluate with your own table

```bash
cargo run --release -p boshiamy_eval -- \
  --cin /path/to/your/table.cin --lm tools/lm/data/zhwiki_tw.bslm \
  --sentences tools/lm/data/sentences_tw.txt --max-sentences 2000 --show-failures 20
```

Reports recall on injected substitution-distance-1 typos, false positives on
clean sentences, and per-sentence latency. Tune `--lambda-edit`,
`--lambda-change`, `--min-score-delta`, `--beam` from the command line.
`gen_synthetic_cin.py` produces a random-code table for smoke-testing the
pipeline when no real table is available; its numbers are not meaningful for
the product.

## Notes

- `build_corpus.sh` classifies each sentence as Traditional or Simplified and
  applies OpenCC `t2tw` or `s2twp` respectively; running `s2twp` on Traditional
  text mangles phrases such as 導出→匯出.
- BSD `awk` compares UTF-8 strings incorrectly; the script forces `LC_ALL=C`.
