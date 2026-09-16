//! Offline evaluation tool (PRD "Offline Evaluation Tool").
//!
//! Given a user-supplied `.cin` (never shipped with the repo), a BSLM language
//! model and a file of correct sentences, this tool:
//!   1. maps each sentence to raw codes (first legal code per char),
//!   2. builds a *typo set* by injecting `--errors-per-sentence` legal
//!      substitution-distance-1 typos (the typed char is a real char of the
//!      mistyped code, so the mistake is invisible to a plain table IME),
//!   3. runs the correction engine on both the clean and typo sets,
//!   4. reports recall, wrong-suggestion rate, false-positive rate and latency.
//!
//! Nothing here uploads or stores user data; sentences stay on disk locally.

use boshiamy_core::{
    CinParser, CodeIndex, CorrectionEngine, CorrectionPolicyConfig, NgramModel, ScoringWeights,
    SentenceSession, SessionUnit,
};
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;
use std::time::Instant;

struct Args {
    cin: String,
    lm: String,
    sentences: String,
    max_sentences: usize,
    errors_per_sentence: usize,
    seed: u64,
    weights: ScoringWeights,
    policy: CorrectionPolicyConfig,
    max_candidates: usize,
    show_failures: usize,
    error_kind: String,
    lambda_seg: Option<f64>,
    no_lattice: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        cin: String::new(),
        lm: String::new(),
        sentences: String::new(),
        max_sentences: 2000,
        errors_per_sentence: 1,
        seed: 42,
        weights: ScoringWeights::mvp_defaults(),
        policy: CorrectionPolicyConfig::mvp_defaults(),
        max_candidates: boshiamy_core::DEFAULT_MAX_CANDIDATES_PER_POSITION,
        show_failures: 0,
        error_kind: "mixed".into(),
        lambda_seg: None,
        no_lattice: false,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = argv
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        let num = |v: &str| v.parse::<f64>().map_err(|e| format!("{key}: {e}"));
        match key {
            "--cin" => a.cin = val.clone(),
            "--lm" => a.lm = val.clone(),
            "--sentences" => a.sentences = val.clone(),
            "--max-sentences" => a.max_sentences = num(val)? as usize,
            "--errors-per-sentence" => a.errors_per_sentence = num(val)? as usize,
            "--seed" => a.seed = num(val)? as u64,
            "--lambda-edit" => a.weights.lambda_edit = num(val)?,
            "--lambda-change" => a.weights.lambda_change = num(val)?,
            "--lambda-choice" => a.weights.lambda_choice = num(val)?,
            "--beam" => a.weights.beam_width = num(val)? as usize,
            "--min-score-delta" => a.policy.min_score_delta = num(val)?,
            "--max-changed" => a.policy.max_changed_absolute = num(val)? as usize,
            "--max-candidates" => a.max_candidates = num(val)? as usize,
            "--show-failures" => a.show_failures = num(val)? as usize,
            "--error-kind" => a.error_kind = val.clone(),
            "--lambda-seg" => a.lambda_seg = Some(num(val)?),
            "--no-lattice" => {
                a.no_lattice = val == "1" || val == "true";
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if a.cin.is_empty() || a.lm.is_empty() || a.sentences.is_empty() {
        return Err("usage: boshiamy-eval --cin table.cin --lm model.bslm --sentences clean.txt [--max-sentences N] [--errors-per-sentence N] [--seed N] [--lambda-edit X] [--lambda-change X] [--lambda-choice X] [--beam N] [--min-score-delta X] [--max-changed N] [--max-candidates N] [--show-failures N] [--error-kind sub|seg|mixed] [--lambda-seg X] [--no-lattice 1]".into());
    }
    Ok(a)
}

/// Small deterministic PRNG (xorshift64*) so runs are reproducible without deps.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn first_code(index: &CodeIndex, ch: char) -> Option<String> {
    index
        .codes_for_char(ch)
        .min_by_key(|c| c.len())
        .map(|s| s.to_string())
}

/// Move one boundary by one letter between units i and i+1 (space pressed early or
/// late). Returns the two replacement (char, code) pairs when both new codes are legal.
fn inject_segmentation_typo(
    index: &CodeIndex,
    rng: &mut Rng,
    left: &(char, String),
    right: &(char, String),
) -> Option<((char, String), (char, String))> {
    let (lc, rc) = (&left.1, &right.1);
    let mut opts: Vec<(String, String)> = Vec::new();
    if lc.len() >= 2 {
        // last letter of left moves to the front of right
        opts.push((
            lc[..lc.len() - 1].to_string(),
            format!("{}{}", &lc[lc.len() - 1..], rc),
        ));
    }
    if rc.len() >= 2 {
        // first letter of right moves to the end of left
        opts.push((format!("{}{}", lc, &rc[..1]), rc[1..].to_string()));
    }
    let mut legal: Vec<((char, String), (char, String))> = Vec::new();
    for (nl, nr) in opts {
        let lch = index.chars_for_code(&nl).next();
        let rch = index.chars_for_code(&nr).next();
        if let (Some(l), Some(r)) = (lch, rch) {
            if l != left.0 || r != right.0 {
                legal.push(((l, nl), (r, nr)));
            }
        }
    }
    if legal.is_empty() {
        return None;
    }
    Some(legal[rng.below(legal.len())].clone())
}

/// Replace one letter of `code` so the result is a legal code producing a different char.
fn inject_typo(
    index: &CodeIndex,
    rng: &mut Rng,
    original: char,
    code: &str,
) -> Option<(char, String)> {
    let neighbors = index.substitution_neighbors(code);
    let pool: Vec<_> = neighbors
        .into_iter()
        .filter(|(_, ch)| *ch != original)
        .collect();
    if pool.is_empty() {
        return None;
    }
    let (c, ch) = pool[rng.below(pool.len())];
    Some((ch, c.to_string()))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let table = match fs::read_to_string(&args.cin)
        .map_err(|e| e.to_string())
        .and_then(|t| CinParser::parse(&t).map_err(|e| e.to_string()))
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cin: {e}");
            return ExitCode::from(1);
        }
    };
    let index = CodeIndex::from_cin_table(&table);
    let t0 = Instant::now();
    let lm = match NgramModel::load(&args.lm) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("lm: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "table: {} codes / {} chars; lm: vocab={} bigrams={} trigrams={} loaded in {:?}",
        index.code_count(),
        index.char_count(),
        lm.vocab_size(),
        lm.bigram_count(),
        lm.trigram_count(),
        t0.elapsed()
    );
    let mut engine = CorrectionEngine::new(
        index,
        Box::new(lm),
        args.weights,
        args.policy,
        args.max_candidates,
    );
    engine.use_lattice = !args.no_lattice;
    if let Some(ls) = args.lambda_seg {
        engine.lattice_weights_mut().lambda_seg = ls;
    }
    let index = engine.index();

    // Load sentences; keep only those fully covered by the table (so typos are always injectable).
    let file = match fs::File::open(&args.sentences) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("sentences: {e}");
            return ExitCode::from(1);
        }
    };
    let mut clean: Vec<Vec<(char, String)>> = Vec::new();
    let mut skipped = 0usize;
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut units = Vec::new();
        let mut ok = true;
        for ch in line.chars() {
            match first_code(index, ch) {
                Some(code) => units.push((ch, code)),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && units.len() >= 4 {
            clean.push(units);
        } else {
            skipped += 1;
        }
        if clean.len() >= args.max_sentences {
            break;
        }
    }
    eprintln!(
        "sentences: {} usable, {} skipped (chars missing from table or too short)",
        clean.len(),
        skipped
    );
    if clean.is_empty() {
        return ExitCode::from(1);
    }

    let mut rng = Rng(args.seed | 1);
    let mut latencies: Vec<f64> = Vec::new();

    // Clean set → false positives.
    let mut false_pos = 0usize;
    let mut fp_examples: Vec<(String, String)> = Vec::new();
    for units in &clean {
        let session = SentenceSession::from_units(
            units
                .iter()
                .map(|(c, k)| SessionUnit::new(*c, k.clone()))
                .collect(),
        );
        let t = Instant::now();
        let s = engine.suggest(&session);
        latencies.push(t.elapsed().as_secs_f64() * 1000.0);
        if let Some(s) = s {
            false_pos += 1;
            if fp_examples.len() < args.show_failures {
                fp_examples.push((session.original_text.clone(), s.corrected_text));
            }
        }
    }

    // Typo set → recall.
    let mut n_typo = 0usize;
    let mut hit = 0usize;
    let mut wrong = 0usize;
    let mut miss = 0usize;
    let mut examples: Vec<(String, String, String)> = Vec::new();
    for units in &clean {
        let mut typed = units.clone();
        let mut injected = 0;
        let mut tries = 0;
        while injected < args.errors_per_sentence && tries < 20 {
            tries += 1;
            let pos = rng.below(typed.len());
            if typed[pos].0 != units[pos].0
                || !boshiamy_core::candidate_generator::is_cjk(units[pos].0)
            {
                continue;
            }
            let use_seg = match args.error_kind.as_str() {
                "seg" => true,
                "sub" => false,
                _ => rng.below(2) == 0,
            };
            if use_seg {
                if pos + 1 >= typed.len()
                    || typed[pos + 1].0 != units[pos + 1].0
                    || !boshiamy_core::candidate_generator::is_cjk(units[pos + 1].0)
                {
                    continue;
                }
                if let Some((l, r)) =
                    inject_segmentation_typo(index, &mut rng, &units[pos], &units[pos + 1])
                {
                    typed[pos] = l;
                    typed[pos + 1] = r;
                    injected += 1;
                }
            } else if let Some((ch, code)) =
                inject_typo(index, &mut rng, units[pos].0, &units[pos].1)
            {
                typed[pos] = (ch, code);
                injected += 1;
            }
        }
        if injected == 0 {
            continue;
        }
        n_typo += 1;
        let intended: String = units.iter().map(|(c, _)| *c).collect();
        let session = SentenceSession::from_units(
            typed
                .iter()
                .map(|(c, k)| SessionUnit::new(*c, k.clone()))
                .collect(),
        );
        let t = Instant::now();
        let s = engine.suggest(&session);
        latencies.push(t.elapsed().as_secs_f64() * 1000.0);
        match s {
            Some(s) if s.corrected_text == intended => hit += 1,
            Some(s) => {
                wrong += 1;
                if examples.len() < args.show_failures {
                    examples.push((session.original_text.clone(), intended, s.corrected_text));
                }
            }
            None => {
                miss += 1;
                if examples.len() < args.show_failures {
                    examples.push((session.original_text.clone(), intended, "(none)".into()));
                }
            }
        }
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_clean = clean.len();
    println!("clean_sentences\t{n_clean}");
    println!(
        "false_positive_rate\t{:.4}\t({false_pos}/{n_clean})",
        false_pos as f64 / n_clean as f64
    );
    println!("typo_sentences\t{n_typo}");
    println!(
        "recall\t{:.4}\t({hit}/{n_typo})",
        hit as f64 / n_typo.max(1) as f64
    );
    println!(
        "wrong_suggestion_rate\t{:.4}\t({wrong}/{n_typo})",
        wrong as f64 / n_typo.max(1) as f64
    );
    println!(
        "no_suggestion_rate\t{:.4}\t({miss}/{n_typo})",
        miss as f64 / n_typo.max(1) as f64
    );
    println!("latency_ms_p50\t{:.2}", percentile(&latencies, 0.5));
    println!("latency_ms_p95\t{:.2}", percentile(&latencies, 0.95));
    println!("latency_ms_max\t{:.2}", percentile(&latencies, 1.0));
    if !fp_examples.is_empty() {
        println!("\n# false positives (original -> suggested)");
        for (o, s) in fp_examples {
            println!("{o}\t{s}");
        }
    }
    if !examples.is_empty() {
        println!("\n# typo failures (typed -> intended -> suggested)");
        for (t, i, s) in examples {
            println!("{t}\t{i}\t{s}");
        }
    }
    ExitCode::SUCCESS
}
