//! Keystroke-lattice decoder with boundary jitter.
//!
//! Phone typos are mostly *segmentation* errors: the space lands one key early
//! or late, so `zya e fa` (等一下) becomes `zya ef a` (等靈對). Every character
//! is then a legal table hit and per-character edits cannot recover it.
//!
//! This decoder keeps the user's units but lets each unit boundary move by at
//! most one letter in either direction. Each unit is then decoded from the
//! letters between its (possibly shifted) boundaries: the exact code's
//! characters (distance 0) plus one-edit neighbours (distance 1, covering an
//! adjacent-key slip, a missed or doubled letter, or two swapped letters).
//! Characters without a raw code (punctuation) are fixed in place. Because
//! every path emits exactly one character per unit, hypotheses are directly
//! comparable and there is no length bias. Paths are scored with the trigram
//! LM minus penalties for edits, changed characters and moved boundaries; the
//! user's original segmentation is one path with zero penalty.

use crate::boshiamy_distance::BoshiamyDistance;
use crate::candidate_generator::is_cjk;
use crate::code_index::CodeIndex;
use crate::language_model::LanguageModel;
use crate::sentence_ranker::RankedSentence;
use crate::session::SentenceSession;
use std::collections::HashMap;

/// Weights for the lattice decoder.
#[derive(Debug, Clone, Copy)]
pub struct LatticeWeights {
    /// Per unit of code edit distance.
    pub lambda_edit: f64,
    /// Per output character that differs from what the user got.
    pub lambda_change: f64,
    /// Per unit boundary moved by one letter.
    pub lambda_seg: f64,
    /// Per changed character the user had explicitly selected from a candidate list.
    pub lambda_choice: f64,
    pub beam_width: usize,
    /// Edit-distance-1 alternatives kept per span (exact-code characters are always kept).
    pub max_edit_candidates: usize,
    /// Longest code considered, in letters.
    pub max_code_len: usize,
    /// How far (in letters) a boundary may move. 1 = space one key early or late.
    pub max_shift: usize,
}

impl LatticeWeights {
    pub fn mvp_defaults() -> Self {
        Self {
            lambda_edit: 3.0,
            lambda_change: 4.0,
            lambda_seg: 1.0,
            lambda_choice: 6.0,
            beam_width: 32,
            max_edit_candidates: 48,
            max_code_len: 5,
            max_shift: 1,
        }
    }
}

#[derive(Clone)]
struct State {
    /// Letter position where the next unit starts.
    pos: usize,
    a: u32,
    b: u32,
    chars: Vec<char>,
    total_distance: f64,
    changed: usize,
    changed_explicit: usize,
    moved_boundaries: usize,
    score: f64,
}

struct UnitSpan {
    start: usize,
    end: usize,
    ch: char,
    explicit: bool,
    /// True for characters without a raw code: fixed, boundaries cannot move.
    fixed: bool,
}

pub struct LatticeDecoder {
    pub weights: LatticeWeights,
}

impl LatticeDecoder {
    pub fn new(weights: LatticeWeights) -> Self {
        Self { weights }
    }

    pub fn decode(
        &self,
        index: &CodeIndex,
        session: &SentenceSession,
        lm: &dyn LanguageModel,
    ) -> Vec<RankedSentence> {
        let w = self.weights;
        // Flatten raw codes into one letter stream; fixed characters occupy one slot.
        let mut letters: Vec<u8> = Vec::new();
        let mut units: Vec<UnitSpan> = Vec::new();
        for unit in &session.units {
            let start = letters.len();
            let fixed = unit.raw_code.is_empty();
            if fixed {
                letters.push(0);
            } else {
                letters.extend(unit.raw_code.bytes());
            }
            units.push(UnitSpan {
                start,
                end: letters.len(),
                ch: unit.output_character,
                explicit: unit.is_explicit_selection(),
                fixed,
            });
        }
        let n = letters.len();
        if n == 0 {
            return Vec::new();
        }

        let mut beam = vec![State {
            pos: 0,
            a: lm.bos_id(),
            b: lm.bos_id(),
            chars: Vec::new(),
            total_distance: 0.0,
            changed: 0,
            changed_explicit: 0,
            moved_boundaries: 0,
            score: 0.0,
        }];
        let mut span_cache: HashMap<(usize, usize), Vec<(char, f64)>> = HashMap::new();

        for (i, unit) in units.iter().enumerate() {
            let is_last = i + 1 == units.len();
            let next_fixed = units.get(i + 1).is_some_and(|u| u.fixed);
            let mut next: Vec<State> = Vec::new();

            for st in &beam {
                if unit.fixed {
                    if st.pos != unit.start {
                        continue;
                    }
                    let c = lm.char_id(unit.ch);
                    let mut chars = st.chars.clone();
                    chars.push(unit.ch);
                    next.push(State {
                        pos: unit.end,
                        a: st.b,
                        b: c,
                        chars,
                        score: st.score + lm.logprob_ids(st.a, st.b, c),
                        ..st.clone()
                    });
                    continue;
                }
                let p = st.pos;
                // End boundary may shift unless it is the sentence end or touches a fixed char.
                let shift = if is_last || next_fixed {
                    0
                } else {
                    w.max_shift
                };
                let lo = unit.end.saturating_sub(shift).max(p + 1);
                let hi = (unit.end + shift).min(n);
                for q in lo..=hi {
                    if q - p > w.max_code_len {
                        continue;
                    }
                    if letters[p..q].contains(&0) {
                        continue;
                    }
                    let cands = span_cache.entry((p, q)).or_insert_with(|| {
                        let code: String = letters[p..q].iter().map(|&b| b as char).collect();
                        let mut cands: Vec<(char, f64)> = index
                            .chars_for_code(&code)
                            .filter(|c| is_cjk(*c))
                            .map(|c| (c, 0.0))
                            .collect();
                        let mut edits: Vec<(char, f64, f64)> = index
                            .edit_neighbors(&code)
                            .into_iter()
                            .filter(|(_, ch, _)| is_cjk(*ch))
                            .map(|(_, ch, kind)| {
                                (ch, BoshiamyDistance::edit_cost(kind), lm.unigram(ch))
                            })
                            .collect();
                        edits.sort_by(|x, y| {
                            y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        edits.dedup_by(|x, y| x.0 == y.0);
                        edits.truncate(w.max_edit_candidates);
                        for (ch, d, _) in edits {
                            if !cands.iter().any(|(c, _)| *c == ch) {
                                cands.push((ch, d));
                            }
                        }
                        cands
                    });
                    let own_span = p == unit.start && q == unit.end;
                    let moved = usize::from(q != unit.end);
                    let mut saw_original = false;
                    for &(ch, dist) in cands.iter() {
                        let is_original = own_span && ch == unit.ch;
                        saw_original |= is_original;
                        push_state(&mut next, st, unit, q, ch, dist, moved, is_original, &w, lm);
                    }
                    // The user's own character on its own span is always a candidate.
                    if own_span && !saw_original {
                        push_state(&mut next, st, unit, q, unit.ch, 0.0, moved, true, &w, lm);
                    }
                }
            }
            if next.is_empty() {
                return Vec::new();
            }
            next.sort_by(|x, y| {
                y.score
                    .partial_cmp(&x.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next.dedup_by(|x, y| x.pos == y.pos && x.chars == y.chars);
            next.truncate(w.beam_width.max(1));
            beam = next;
        }

        let mut ranked: Vec<RankedSentence> = beam
            .into_iter()
            .filter(|st| st.pos == n)
            .map(|st| {
                let score = st.score + lm.logprob_ids(st.a, st.b, lm.eos_id());
                let language_score = score
                    + w.lambda_edit * st.total_distance
                    + w.lambda_change * st.changed as f64
                    + w.lambda_seg * st.moved_boundaries as f64
                    + w.lambda_choice * st.changed_explicit as f64;
                RankedSentence {
                    text: st.chars.iter().collect(),
                    chars: st.chars,
                    total_distance: st.total_distance,
                    changed_count: st.changed,
                    changed_explicit_selections: st.changed_explicit,
                    language_score,
                    score,
                }
            })
            .collect();
        ranked.sort_by(|x, y| {
            y.score
                .partial_cmp(&x.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.dedup_by(|x, y| x.text == y.text);
        ranked
    }
}

#[allow(clippy::too_many_arguments)]
fn push_state(
    next: &mut Vec<State>,
    st: &State,
    unit: &UnitSpan,
    q: usize,
    ch: char,
    dist: f64,
    moved: usize,
    is_original: bool,
    w: &LatticeWeights,
    lm: &dyn LanguageModel,
) {
    let changed = usize::from(!is_original);
    let explicit = usize::from(unit.explicit && !is_original);
    let c = lm.char_id(ch);
    let penalty = w.lambda_edit * dist
        + w.lambda_change * changed as f64
        + w.lambda_seg * moved as f64
        + w.lambda_choice * explicit as f64;
    let mut chars = st.chars.clone();
    chars.push(ch);
    next.push(State {
        pos: q,
        a: st.b,
        b: c,
        chars,
        total_distance: st.total_distance + dist,
        changed: st.changed + changed,
        changed_explicit: st.changed_explicit + explicit,
        moved_boundaries: st.moved_boundaries + moved,
        score: st.score + lm.logprob_ids(st.a, st.b, c) - penalty,
    });
}
