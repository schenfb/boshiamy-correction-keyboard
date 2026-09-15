//! Train a character trigram LM with interpolated Kneser-Ney smoothing, prune by
//! count thresholds, renormalize backoff weights, and write `BSLM` v1.
//!
//! Input: one sentence per line (UTF-8). Output: binary model loadable by
//! `boshiamy_core::NgramModel`.

use boshiamy_core::ngram_model::{BOS, EOS, FIRST_CHAR_ID, UNK};
use boshiamy_core::NgramModel;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::ExitCode;

struct Args {
    input: String,
    output: String,
    min_char_count: u32,
    min_bigram: u32,
    min_trigram: u32,
    heldout_every: usize,
    max_sentences: usize,
    max_bigrams: usize,
    max_trigrams: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        input: String::new(),
        output: String::new(),
        min_char_count: 5,
        min_bigram: 3,
        min_trigram: 5,
        heldout_every: 200,
        max_sentences: 0,
        max_bigrams: 0,
        max_trigrams: 0,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = argv
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--input" => a.input = val.clone(),
            "--output" => a.output = val.clone(),
            "--min-char-count" => a.min_char_count = val.parse().map_err(|e| format!("{e}"))?,
            "--min-bigram" => a.min_bigram = val.parse().map_err(|e| format!("{e}"))?,
            "--min-trigram" => a.min_trigram = val.parse().map_err(|e| format!("{e}"))?,
            "--heldout-every" => a.heldout_every = val.parse().map_err(|e| format!("{e}"))?,
            "--max-sentences" => a.max_sentences = val.parse().map_err(|e| format!("{e}"))?,
            "--max-bigrams" => a.max_bigrams = val.parse().map_err(|e| format!("{e}"))?,
            "--max-trigrams" => a.max_trigrams = val.parse().map_err(|e| format!("{e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if a.input.is_empty() || a.output.is_empty() {
        return Err("usage: boshiamy-lm-train --input sentences.txt --output model.bslm [--min-char-count N] [--min-bigram N] [--min-trigram N] [--heldout-every N] [--max-sentences N] [--max-bigrams N] [--max-trigrams N]".into());
    }
    Ok(a)
}

#[inline]
fn bi_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}
#[inline]
fn tri_key(a: u32, b: u32, c: u32) -> u64 {
    ((a as u64) << 40) | ((b as u64) << 20) | c as u64
}
#[inline]
fn tri_split(k: u64) -> (u32, u32, u32) {
    (
        (k >> 40) as u32,
        ((k >> 20) & 0xFFFFF) as u32,
        (k & 0xFFFFF) as u32,
    )
}
#[inline]
fn bi_split(k: u64) -> (u32, u32) {
    ((k >> 32) as u32, (k & 0xFFFF_FFFF) as u32)
}

/// Keep only the `max` most frequent keys (0 = unlimited).
fn cap_by_count(keys: &mut Vec<u64>, counts: &HashMap<u64, u32>, max: usize) {
    if max > 0 && keys.len() > max {
        keys.sort_unstable_by_key(|k| std::cmp::Reverse(counts[k]));
        keys.truncate(max);
    }
}

fn is_heldout(line_no: usize, every: usize) -> bool {
    every > 0 && line_no % every == 0
}

fn discount(n1: u64, n2: u64) -> f64 {
    if n1 == 0 || n1 + 2 * n2 == 0 {
        return 0.5;
    }
    (n1 as f64 / (n1 as f64 + 2.0 * n2 as f64)).clamp(0.1, 0.95)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    // Pass 1: character counts → vocab.
    eprintln!("pass 1: counting characters");
    let mut char_counts: HashMap<char, u32> = HashMap::new();
    let mut n_lines = 0usize;
    {
        let f = match File::open(&args.input) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("open {}: {e}", args.input);
                return ExitCode::from(1);
            }
        };
        for (i, line) in BufReader::new(f).lines().enumerate() {
            let Ok(line) = line else { break };
            if args.max_sentences > 0 && i >= args.max_sentences {
                break;
            }
            n_lines += 1;
            if is_heldout(i, args.heldout_every) {
                continue;
            }
            for ch in line.chars() {
                *char_counts.entry(ch).or_insert(0) += 1;
            }
        }
    }
    let mut vocab: Vec<char> = char_counts
        .iter()
        .filter(|(_, &n)| n >= args.min_char_count)
        .map(|(&c, _)| c)
        .collect();
    vocab.sort_unstable();
    let char_to_id: HashMap<char, u32> = vocab
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, FIRST_CHAR_ID + i as u32))
        .collect();
    let v_total = vocab.len() + FIRST_CHAR_ID as usize;
    if v_total >= (1 << 20) {
        eprintln!("vocab too large for 20-bit ids: {v_total}");
        return ExitCode::from(1);
    }
    eprintln!(
        "  lines={n_lines} distinct_chars={} vocab={}",
        char_counts.len(),
        vocab.len()
    );
    drop(char_counts);

    // Pass 2: n-gram counts.
    eprintln!("pass 2: counting n-grams");
    let mut uni: Vec<u64> = vec![0; v_total];
    let mut bi: HashMap<u64, u32> = HashMap::new();
    let mut tri: HashMap<u64, u32> = HashMap::new();
    let mut heldout: Vec<Vec<u32>> = Vec::new();
    let mut total_tokens = 0u64;
    {
        let f = File::open(&args.input).unwrap();
        let mut ids: Vec<u32> = Vec::new();
        for (i, line) in BufReader::new(f).lines().enumerate() {
            let Ok(line) = line else { break };
            if args.max_sentences > 0 && i >= args.max_sentences {
                break;
            }
            ids.clear();
            ids.extend(
                line.chars()
                    .map(|c| char_to_id.get(&c).copied().unwrap_or(UNK)),
            );
            if is_heldout(i, args.heldout_every) {
                heldout.push(ids.clone());
                continue;
            }
            let mut a = BOS;
            let mut b = BOS;
            *bi.entry(bi_key(BOS, BOS)).or_insert(0) += 1;
            for &c in ids.iter().chain(std::iter::once(&EOS)) {
                uni[c as usize] += 1;
                *bi.entry(bi_key(b, c)).or_insert(0) += 1;
                *tri.entry(tri_key(a, b, c)).or_insert(0) += 1;
                a = b;
                b = c;
                total_tokens += 1;
            }
            if i % 2_000_000 == 0 && i > 0 {
                eprintln!("  {i} lines, bigrams={} trigrams={}", bi.len(), tri.len());
            }
        }
    }
    eprintln!(
        "  tokens={total_tokens} bigrams={} trigrams={} heldout={}",
        bi.len(),
        tri.len(),
        heldout.len()
    );

    // Count-of-counts for discounts.
    let (mut n1_3, mut n2_3, mut n1_2, mut n2_2) = (0u64, 0u64, 0u64, 0u64);
    for &n in tri.values() {
        if n == 1 {
            n1_3 += 1
        } else if n == 2 {
            n2_3 += 1
        }
    }
    for &n in bi.values() {
        if n == 1 {
            n1_2 += 1
        } else if n == 2 {
            n2_2 += 1
        }
    }
    let d3 = discount(n1_3, n2_3);
    let d2 = discount(n1_2, n2_2);
    eprintln!("  discounts d3={d3:.3} d2={d2:.3}");

    // Context totals for trigrams (unpruned): n(ab.) and N1+(ab.)
    let mut ctx_total: HashMap<u64, u64> = HashMap::new(); // key bi(a,b)
    let mut ctx_types: HashMap<u64, u32> = HashMap::new();
    // Continuation counts for bigram level: N1+(.bc) keyed bi(b,c); N1+(.b.) and N1+(b.) keyed b
    let mut cont_bc: HashMap<u64, u32> = HashMap::new();
    for (&k, &n) in tri.iter() {
        let (a, b, c) = tri_split(k);
        let ab = bi_key(a, b);
        *ctx_total.entry(ab).or_insert(0) += n as u64;
        *ctx_types.entry(ab).or_insert(0) += 1;
        *cont_bc.entry(bi_key(b, c)).or_insert(0) += 1;
    }
    // For contexts b == BOS use raw counts (BOS is never predicted, so continuation counts degenerate).
    let mut cont_b_total: Vec<u64> = vec![0; v_total]; // N1+(.b.) or raw n(b.) for BOS
    let mut cont_b_types: Vec<u32> = vec![0; v_total]; // N1+(b.)
    let mut cont_c: Vec<u64> = vec![0; v_total]; // N1+(.c) for unigram
    for (&k, &n) in bi.iter() {
        let (b, c) = bi_split(k);
        let cnt = if b == BOS {
            n as u64
        } else {
            cont_bc.get(&k).copied().unwrap_or(0) as u64
        };
        cont_b_total[b as usize] += cnt;
        cont_b_types[b as usize] += 1;
        if b != BOS || c != BOS {
            cont_c[c as usize] += 1;
        }
    }
    let cont_all: u64 = cont_c.iter().sum();

    // Unigram probabilities (continuation counts, additive smoothing for unseen/UNK).
    let vf = v_total as f64;
    let mut uni_lp: Vec<f32> = vec![0.0; v_total];
    for c in 0..v_total {
        let p = (cont_c[c] as f64 + 1.0) / (cont_all as f64 + vf);
        uni_lp[c] = p.ln() as f32;
    }
    uni_lp[BOS as usize] = -99.0;
    let unk_lp = uni_lp[UNK as usize];

    // Bigram probabilities (pruned) + unigram backoff weights.
    let mut bi_keys: Vec<u64> = bi
        .iter()
        .filter(|(&k, &n)| {
            let (b, c) = bi_split(k);
            n >= args.min_bigram && c != BOS && !(b == BOS && c == EOS)
        })
        .map(|(&k, _)| k)
        .collect();
    cap_by_count(&mut bi_keys, &bi, args.max_bigrams);
    bi_keys.sort_unstable();
    let mut bi_lp: Vec<f32> = Vec::with_capacity(bi_keys.len());
    // Interpolated KN bigram prob (unpruned interpolation weight γ(b)).
    let gamma_b = |b: u32| -> f64 {
        let tot = cont_b_total[b as usize] as f64;
        if tot == 0.0 {
            1.0
        } else {
            d2 * cont_b_types[b as usize] as f64 / tot
        }
    };
    for &k in &bi_keys {
        let (b, c) = bi_split(k);
        let n = bi[&k];
        let cnt = if b == BOS {
            n as f64
        } else {
            cont_bc.get(&k).copied().unwrap_or(0) as f64
        };
        let tot = cont_b_total[b as usize] as f64;
        let p = ((cnt - d2).max(0.0) / tot) + gamma_b(b) * (uni_lp[c as usize] as f64).exp();
        bi_lp.push(p.ln() as f32);
    }
    // Backoff weights for unigram contexts b: (1 - Σ_kept p(c|b)) / (1 - Σ_kept p1(c)).
    let mut uni_bo: Vec<f32> = vec![0.0; v_total];
    {
        let mut sum_hi: Vec<f64> = vec![0.0; v_total];
        let mut sum_lo: Vec<f64> = vec![0.0; v_total];
        for (i, &k) in bi_keys.iter().enumerate() {
            let (b, c) = bi_split(k);
            sum_hi[b as usize] += (bi_lp[i] as f64).exp();
            sum_lo[b as usize] += (uni_lp[c as usize] as f64).exp();
        }
        for b in 0..v_total {
            let num = (1.0 - sum_hi[b]).max(1e-9);
            let den = (1.0 - sum_lo[b]).max(1e-9);
            uni_bo[b] = (num / den).ln() as f32;
        }
    }
    let bi_index: HashMap<u64, usize> = bi_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let lp2 = |b: u32, c: u32| -> f64 {
        match bi_index.get(&bi_key(b, c)) {
            Some(&i) => bi_lp[i] as f64,
            None => uni_bo[b as usize] as f64 + uni_lp[c as usize] as f64,
        }
    };

    // Trigram probabilities (pruned) + bigram backoff weights.
    let mut tri_keys: Vec<u64> = tri
        .iter()
        .filter(|(_, &n)| n >= args.min_trigram)
        .map(|(&k, _)| k)
        .collect();
    cap_by_count(&mut tri_keys, &tri, args.max_trigrams);
    tri_keys.sort_unstable();
    let mut tri_lp: Vec<f32> = Vec::with_capacity(tri_keys.len());
    for &k in &tri_keys {
        let (a, b, c) = tri_split(k);
        let n = tri[&k] as f64;
        let ab = bi_key(a, b);
        let tot = ctx_total[&ab] as f64;
        let gamma = d3 * ctx_types[&ab] as f64 / tot;
        let p = ((n - d3).max(0.0) / tot) + gamma * lp2(b, c).exp();
        tri_lp.push(p.ln() as f32);
    }
    let mut bi_bo: Vec<f32> = vec![0.0; bi_keys.len()];
    {
        let mut sum_hi: HashMap<u64, f64> = HashMap::new();
        let mut sum_lo: HashMap<u64, f64> = HashMap::new();
        for (i, &k) in tri_keys.iter().enumerate() {
            let (a, b, c) = tri_split(k);
            let ab = bi_key(a, b);
            *sum_hi.entry(ab).or_insert(0.0) += (tri_lp[i] as f64).exp();
            *sum_lo.entry(ab).or_insert(0.0) += lp2(b, c).exp();
        }
        for (i, &k) in bi_keys.iter().enumerate() {
            let hi = sum_hi.get(&k).copied().unwrap_or(0.0);
            let lo = sum_lo.get(&k).copied().unwrap_or(0.0);
            bi_bo[i] = ((1.0 - hi).max(1e-9) / (1.0 - lo).max(1e-9)).ln() as f32;
        }
    }
    eprintln!(
        "  kept bigrams={} trigrams={}",
        bi_keys.len(),
        tri_keys.len()
    );

    // Write BSLM v1.
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"BSLM");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
    for &c in &vocab {
        bytes.extend_from_slice(&(c as u32).to_le_bytes());
    }
    bytes.extend_from_slice(&unk_lp.to_le_bytes());
    for &x in &uni_lp {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    for &x in &uni_bo {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes.extend_from_slice(&(bi_keys.len() as u32).to_le_bytes());
    for &x in &bi_keys {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    for &x in &bi_lp {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    for &x in &bi_bo {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes.extend_from_slice(&(tri_keys.len() as u32).to_le_bytes());
    for &x in &tri_keys {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    for &x in &tri_lp {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    {
        let mut w = BufWriter::new(File::create(&args.output).unwrap());
        w.write_all(&bytes).unwrap();
    }
    eprintln!("wrote {} ({:.1} MB)", args.output, bytes.len() as f64 / 1e6);

    // Held-out perplexity using the real loader (validates format + smoothing).
    if !heldout.is_empty() {
        let model = NgramModel::from_bytes(&bytes).expect("reload model");
        let mut lp = 0.0f64;
        let mut n = 0u64;
        let mut oov = 0u64;
        for ids in &heldout {
            lp += model.score_ids(ids) as f64;
            n += ids.len() as u64 + 1;
            oov += ids.iter().filter(|&&c| c == UNK).count() as u64;
        }
        eprintln!(
            "heldout: sentences={} tokens={n} oov_rate={:.4} perplexity={:.2}",
            heldout.len(),
            oov as f64 / n as f64,
            (-lp / n as f64).exp()
        );
    }
    ExitCode::SUCCESS
}
