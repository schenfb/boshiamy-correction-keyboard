//! Language scoring hook + tiny synthetic Traditional Chinese n-gram stub.
//!
//! The stub is **invented for demos/tests only**. It is not derived from any
//! proprietary Boshiamy data or large copyrighted corpus. Replace via
//! [`LanguageModel`] when a real offline open-license model is available.

use std::collections::HashMap;

/// Pluggable language model interface for sentence scoring.
pub trait LanguageModel: Send + Sync {
    /// Higher is better. Implementations should be offline-only.
    fn score_sentence(&self, text: &str) -> f64;

    /// Context-free prior for a single character, used to rank capped candidate
    /// lists. Default approximates it with a sentence-initial transition.
    fn unigram(&self, ch: char) -> f64 {
        self.score_transition("", ch)
    }

    /// Optional incremental score when appending `next` after `prefix`.
    /// Default falls back to full-sentence scoring.
    fn score_transition(&self, prefix: &str, next: char) -> f64 {
        let mut s = String::with_capacity(prefix.len() + next.len_utf8());
        s.push_str(prefix);
        s.push(next);
        self.score_sentence(&s)
    }
}

/// Minimal character bigram / phrase-boost stub for Traditional Chinese demos.
#[derive(Debug, Clone)]
pub struct StubNgramModel {
    /// log-weight for character bigrams (prev, next) → score contribution.
    bigrams: HashMap<(char, char), f64>,
    /// Bonus when a known good phrase substring appears.
    phrase_boosts: Vec<(String, f64)>,
    /// Unigram prior.
    unigrams: HashMap<char, f64>,
    default_unigram: f64,
    default_bigram: f64,
}

impl StubNgramModel {
    pub fn new(
        bigrams: HashMap<(char, char), f64>,
        phrase_boosts: Vec<(String, f64)>,
        unigrams: HashMap<char, f64>,
        default_unigram: f64,
        default_bigram: f64,
    ) -> Self {
        Self {
            bigrams,
            phrase_boosts,
            unigrams,
            default_unigram,
            default_bigram,
        }
    }

    /// Tiny synthetic model tuned so AC2 prefers 「如果偶爾」 over 「甘果側而」.
    ///
    /// License: original stub data under Apache-2.0 (this repository).
    /// Hook for a real model: implement [`LanguageModel`] and pass into
    /// [`crate::CorrectionEngine::new`].
    pub fn default_traditional_chinese_stub() -> Self {
        let mut bigrams = HashMap::new();
        let pairs = [
            ('這', '樣', 2.0),
            ('樣', '如', 3.5),
            ('如', '果', 5.0),
            ('果', '偶', 3.5),
            ('偶', '爾', 5.0),
            ('爾', '打', 2.5),
            ('打', '錯', 4.0),
            ('錯', '一', 2.0),
            ('一', '個', 4.0),
            ('個', '字', 3.0),
            ('字', '也', 2.0),
            ('也', '沒', 3.0),
            ('沒', '關', 4.0),
            ('關', '係', 5.0),
            // Weak / negative-ish paths for the typo sequence
            ('樣', '甘', 0.2),
            ('甘', '果', 0.3),
            ('果', '側', 0.2),
            ('側', '而', 0.2),
            ('而', '打', 0.5),
        ];
        for (a, b, w) in pairs {
            bigrams.insert((a, b), w);
        }

        let phrase_boosts = vec![
            ("這樣".into(), 2.0),
            ("如果".into(), 6.0),
            ("偶爾".into(), 6.0),
            ("如果偶爾".into(), 4.0),
            ("打錯".into(), 2.0),
            ("一個字".into(), 3.0),
            ("沒關係".into(), 4.0),
            ("甘果".into(), -2.0),
            ("側而".into(), -2.0),
            ("甘果側而".into(), -4.0),
        ];

        let mut unigrams = HashMap::new();
        for ch in "這樣如果偶爾打錯一個字也沒關係甘側而".chars() {
            unigrams.insert(ch, 1.0);
        }

        Self::new(bigrams, phrase_boosts, unigrams, 0.1, 0.05)
    }
}

impl LanguageModel for StubNgramModel {
    fn score_sentence(&self, text: &str) -> f64 {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        for (i, &ch) in chars.iter().enumerate() {
            score += self
                .unigrams
                .get(&ch)
                .copied()
                .unwrap_or(self.default_unigram);
            if i > 0 {
                let prev = chars[i - 1];
                score += self
                    .bigrams
                    .get(&(prev, ch))
                    .copied()
                    .unwrap_or(self.default_bigram);
            }
        }

        for (phrase, boost) in &self.phrase_boosts {
            if text.contains(phrase) {
                score += boost;
            }
        }

        score
    }

    fn score_transition(&self, prefix: &str, next: char) -> f64 {
        let uni = self
            .unigrams
            .get(&next)
            .copied()
            .unwrap_or(self.default_unigram);
        let bi = match prefix.chars().last() {
            Some(prev) => self
                .bigrams
                .get(&(prev, next))
                .copied()
                .unwrap_or(self.default_bigram),
            None => 0.0,
        };
        // Local transition score used by beam search (not full phrase rescoring).
        uni + bi
    }
}

/// Hook documentation type: swap this for a real offline model later.
pub type LanguageModelHook = Box<dyn LanguageModel>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_prefers_correct_phrase() {
        let lm = StubNgramModel::default_traditional_chinese_stub();
        let good = lm.score_sentence("這樣如果偶爾打錯一個字也沒關係");
        let bad = lm.score_sentence("這樣甘果側而打錯一個字也沒關係");
        assert!(good > bad, "good={good} bad={bad}");
    }
}
