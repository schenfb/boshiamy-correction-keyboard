//! Boshiamy-compatible whole-sentence correction Shared Core (Phase 1).
//!
//! This crate is fully offline and never embeds a real 嘸蝦米 / Boshiamy code table.
//! Callers supply their own `.cin` data (or synthetic fixtures for tests).

pub mod boshiamy_distance;
pub mod candidate_generator;
pub mod cin_parser;
pub mod code_index;
pub mod correction_policy;
pub mod language_model;
pub mod lattice;
pub mod ngram_model;
pub mod sentence_ranker;
pub mod session;

pub use boshiamy_distance::BoshiamyDistance;
pub use candidate_generator::{CandidateGenerator, PositionCandidate};
pub use cin_parser::{CinParseError, CinParser, CinTable};
pub use code_index::{CodeIndex, EditKind};
pub use correction_policy::{CorrectionPolicy, CorrectionPolicyConfig, CorrectionSuggestion};
pub use language_model::{LanguageModel, StubNgramModel};
pub use lattice::{LatticeDecoder, LatticeWeights};
pub use ngram_model::{NgramLoadError, NgramModel};
pub use sentence_ranker::{RankedSentence, ScoringWeights, SentenceRanker};
pub use session::{SentenceSession, SessionUnit};

/// Per-position candidate cap. Recall is bounded by whether the intended character
/// survives this cap; 32 recovered most of the headroom in offline evaluation at
/// ~5 ms p95 per sentence, versus 19% recall at 8.
pub const DEFAULT_MAX_CANDIDATES_PER_POSITION: usize = 32;

/// High-level correction engine wiring index + LM + policy.
pub struct CorrectionEngine {
    index: CodeIndex,
    language_model: Box<dyn LanguageModel>,
    generator: CandidateGenerator,
    ranker: SentenceRanker,
    lattice: LatticeDecoder,
    policy: CorrectionPolicy,
    /// Re-segment the keystroke stream (handles misplaced spaces) instead of
    /// per-character substitution only.
    pub use_lattice: bool,
}

impl CorrectionEngine {
    pub fn new(
        index: CodeIndex,
        language_model: Box<dyn LanguageModel>,
        weights: ScoringWeights,
        policy: CorrectionPolicyConfig,
        max_candidates_per_position: usize,
    ) -> Self {
        Self {
            index,
            language_model,
            generator: CandidateGenerator::new(max_candidates_per_position),
            ranker: SentenceRanker::new(weights),
            lattice: LatticeDecoder::new(LatticeWeights {
                lambda_edit: weights.lambda_edit,
                lambda_change: weights.lambda_change,
                lambda_choice: weights.lambda_choice,
                beam_width: weights.beam_width,
                ..LatticeWeights::mvp_defaults()
            }),
            policy: CorrectionPolicy::new(policy),
            use_lattice: true,
        }
    }

    /// Build an engine with default MVP weights/policy and the stub LM.
    pub fn with_defaults(index: CodeIndex) -> Self {
        Self::new(
            index,
            Box::new(StubNgramModel::default_traditional_chinese_stub()),
            ScoringWeights::mvp_defaults(),
            CorrectionPolicyConfig::mvp_defaults(),
            DEFAULT_MAX_CANDIDATES_PER_POSITION,
        )
    }

    pub fn index(&self) -> &CodeIndex {
        &self.index
    }

    pub fn lattice_weights_mut(&mut self) -> &mut LatticeWeights {
        &mut self.lattice.weights
    }

    /// Suggest a whole-sentence correction, or `None` if policy rejects.
    pub fn suggest(&self, session: &SentenceSession) -> Option<CorrectionSuggestion> {
        if self.policy.should_skip_session(session) {
            return None;
        }
        let ranked = self.rank(session);
        self.policy
            .select_suggestion(session, &ranked, self.language_model.as_ref())
    }

    /// Score of the sentence as typed, on the same scale as [`rank`](Self::rank).
    pub fn original_score(&self, session: &SentenceSession) -> f64 {
        self.language_model.score_sentence(&session.original_text)
    }

    /// All ranked candidate sentences (best first), before the display policy.
    pub fn rank(&self, session: &SentenceSession) -> Vec<RankedSentence> {
        let ranked = if self.use_lattice {
            self.lattice
                .decode(&self.index, session, self.language_model.as_ref())
        } else {
            let per_position =
                self.generator
                    .generate(&self.index, session, self.language_model.as_ref());
            self.ranker
                .rank(session, &per_position, self.language_model.as_ref())
        };
        ranked
    }
}
