//! Beam-search sentence ranker.

use crate::candidate_generator::PositionCandidate;
use crate::language_model::LanguageModel;
use crate::session::SentenceSession;

/// Weights for `languageScore - λ_edit*distance - λ_change*changedCount - λ_choice*changedExplicit`.
#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    pub lambda_edit: f64,
    pub lambda_change: f64,
    pub lambda_choice: f64,
    pub beam_width: usize,
}

impl ScoringWeights {
    pub fn mvp_defaults() -> Self {
        Self {
            // Tuned with boshiamy_eval against the Wikipedia trigram model (natural-log
            // scores): higher change penalties cut wrong suggestions to ~2-5% while
            // keeping false positives on clean text near zero. Retune when the LM changes.
            lambda_edit: 3.0,
            lambda_change: 3.0,
            lambda_choice: 6.0,
            beam_width: 32,
        }
    }
}

/// A fully scored candidate sentence.
#[derive(Debug, Clone)]
pub struct RankedSentence {
    pub text: String,
    pub chars: Vec<char>,
    pub total_distance: f64,
    pub changed_count: usize,
    pub changed_explicit_selections: usize,
    pub language_score: f64,
    pub score: f64,
}

#[derive(Clone)]
struct BeamState {
    chars: Vec<char>,
    total_distance: f64,
    changed_count: usize,
    changed_explicit: usize,
    /// Partial path score using transition LM + penalties so far.
    partial_score: f64,
}

/// Beam search over per-position candidates.
#[derive(Debug, Clone)]
pub struct SentenceRanker {
    pub weights: ScoringWeights,
}

impl SentenceRanker {
    pub fn new(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    pub fn rank(
        &self,
        session: &SentenceSession,
        per_position: &[Vec<PositionCandidate>],
        lm: &dyn LanguageModel,
    ) -> Vec<RankedSentence> {
        assert_eq!(session.len(), per_position.len());

        let mut beam = vec![BeamState {
            chars: Vec::new(),
            total_distance: 0.0,
            changed_count: 0,
            changed_explicit: 0,
            partial_score: 0.0,
        }];

        for (pos, candidates) in per_position.iter().enumerate() {
            let unit = &session.units[pos];
            let mut next_beam: Vec<BeamState> = Vec::new();

            for state in &beam {
                let prefix: String = state.chars.iter().collect();
                for cand in candidates {
                    let changed = cand.character != unit.output_character;
                    let changed_explicit = changed && unit.is_explicit_selection();
                    let transition = lm.score_transition(&prefix, cand.character);
                    let penalty = self.weights.lambda_edit * cand.distance
                        + self.weights.lambda_change * if changed { 1.0 } else { 0.0 }
                        + self.weights.lambda_choice * if changed_explicit { 1.0 } else { 0.0 };

                    let mut chars = state.chars.clone();
                    chars.push(cand.character);
                    next_beam.push(BeamState {
                        chars,
                        total_distance: state.total_distance + cand.distance,
                        changed_count: state.changed_count + usize::from(changed),
                        changed_explicit: state.changed_explicit + usize::from(changed_explicit),
                        partial_score: state.partial_score + transition - penalty,
                    });
                }
            }

            next_beam.sort_by(|a, b| {
                b.partial_score
                    .partial_cmp(&a.partial_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next_beam.truncate(self.weights.beam_width.max(1));
            beam = next_beam;
        }

        let mut ranked: Vec<RankedSentence> = beam
            .into_iter()
            .map(|state| {
                let text: String = state.chars.iter().collect();
                let language_score = lm.score_sentence(&text);
                let score = language_score
                    - self.weights.lambda_edit * state.total_distance
                    - self.weights.lambda_change * state.changed_count as f64
                    - self.weights.lambda_choice * state.changed_explicit as f64;
                RankedSentence {
                    text,
                    chars: state.chars,
                    total_distance: state.total_distance,
                    changed_count: state.changed_count,
                    changed_explicit_selections: state.changed_explicit,
                    language_score,
                    score,
                }
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }
}
