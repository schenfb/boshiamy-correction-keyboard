# Boshiamy Correction Keyboard — Shared Core (Phase 1)

Offline Rust **Shared Core** for a free, no-ads, Apache-2.0 whole-sentence
correction keyboard compatible with Boshiamy-style (嘸蝦米) *table* input.

> **Non-official disclaimer:** This project is **not** affiliated with, endorsed
> by, or connected to the official Boshiamy / 嘸蝦米 product or its copyright
> holders. “Boshiamy-compatible” only describes input *behavior* (letter codes →
> characters via a user-supplied table).

## No real code table policy

- This repository **never** commits, downloads, or embeds a real Boshiamy / 嘸蝦米
  code table.
- Apps must let users import their own `.cin` from local storage.
- Tests and the CLI demo use **tiny synthetic** `.cin` fixtures invented for this
  repo (`crates/boshiamy_core/tests/fixtures/`).

## Architecture

```
boshiamy_core (library)
├── CINParser          UTF-8 .cin → mappings
├── CodeIndex          bidirectional code ↔ chars (multi-code / char)
├── BoshiamyDistance   MVP: substitution only (0 or 1.0)
├── CandidateGenerator original + distance≤1 alts, capped per position
├── SentenceRanker     beam search + scoring
├── CorrectionPolicy   thresholds, skip English/URL/email/password/code/digits
├── LanguageModel      trait; StubNgramModel (tests) and NgramModel (BSLM trigram file)
└── SentenceSession    originalText + units (char, rawCode, selectedIndex, timestamp)

boshiamy_cli (binary: boshiamy-correct)
└── load .cin + session units [+ --lm model.bslm] → print best correction or "none"

boshiamy_lm_train (binary: boshiamy-lm-train)
└── sentences.txt → pruned Kneser-Ney character trigram → .bslm (see tools/lm/)

boshiamy_eval (binary: boshiamy-eval)
└── inject distance-1 typos into clean sentences → recall / false positives / latency
```

Candidate lookup uses a wildcard index (`a?c`) built at import time, so each
position costs O(code length) regardless of table size.

### Scoring

```
score = languageScore
      - λ_edit   * totalBoshiamyDistance
      - λ_change * changedCharacterCount
      - λ_choice * changedExplicitSelections
```

Defaults (tuned with `boshiamy_eval` on the Wikipedia trigram model, natural-log
scores): `λ_edit=3.0`, `λ_change=4.0` (scaled down for rare typed characters), `λ_seg=1.0`, `λ_choice=6.0`, `min_score_delta=2.0`,
32 candidates per position, beam 32.

The real model is a character trigram trained from Chinese Wikipedia
(CC BY-SA 4.0) by the pipeline in `tools/lm/`; see `tools/lm/README.md` and
`THIRD_PARTY_NOTICES`. Any other `LanguageModel` implementation can be passed to
`CorrectionEngine::new`.

### Policy (MVP)

Suggest only when the best candidate is **different**, score delta ≥ threshold,
and changed characters ≤ `min(3, 25% of length)`. Skip pure English, URLs,
emails, password-like strings, code-like strings, and mostly-digit text.

### Distance MVP

| Op           | Cost | Status   |
|--------------|------|----------|
| Identical    | 0    | yes      |
| Substitute 1 | 1.0  | yes      |
| Insert/Delete/Transpose | — | **not** in Phase 1 |

## Build & test

```bash
cargo test
cargo build --release -p boshiamy_cli
```

## AC2 demo (synthetic table)

Synthetic mapping (also documented in the fixture and tests):

| Intended | Code | Typed char | Typed code | Distance |
|----------|------|------------|------------|----------|
| 如       | ba   | 甘         | bb         | 1        |
| 偶       | ca   | 側         | cb         | 1        |
| 爾       | da   | 而         | db         | 1        |

Typed sentence: `這樣甘果側而打錯一個字也沒關係`  
Expected correction: `這樣如果偶爾打錯一個字也沒關係`

```bash
cargo run -p boshiamy_cli -- \
  --cin crates/boshiamy_core/tests/fixtures/synthetic_ac2.cin \
  --units "這:aa,樣:ab,甘:bb,果:bc,側:cb,而:db,打:ea,錯:eb,一:ec,個:ed,字:ee,也:ef,沒:eg,關:eh,係:ei"
```

Expected stdout: `這樣如果偶爾打錯一個字也沒關係`

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

Identifiers in code are English; Traditional Chinese may appear in docs/comments.
