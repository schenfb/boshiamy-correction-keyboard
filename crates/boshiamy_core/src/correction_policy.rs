//! Gate whether a ranked correction should be shown to the user.

use crate::language_model::LanguageModel;
use crate::sentence_ranker::RankedSentence;
use crate::session::SentenceSession;

/// Tunable policy thresholds.
#[derive(Debug, Clone, Copy)]
pub struct CorrectionPolicyConfig {
    /// Minimum `best.score - original.score` required to suggest.
    pub min_score_delta: f64,
    /// Hard cap on changed characters (also combined with percent rule).
    pub max_changed_absolute: usize,
    /// Fraction of sentence length (0.25 = 25%).
    pub max_changed_fraction: f64,
}

impl CorrectionPolicyConfig {
    pub fn mvp_defaults() -> Self {
        Self {
            // Recall-first: the bar is a suggestion the user can ignore, so a small
            // margin is enough. Offline (real table, mixed errors): recall 76%,
            // wrong 15%, false positives 2.3%.
            min_score_delta: 2.0,
            max_changed_absolute: 3,
            max_changed_fraction: 0.25,
        }
    }
}

/// Suggestion returned when policy accepts a correction.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionSuggestion {
    pub corrected_text: String,
    pub original_text: String,
    pub score: f64,
    pub original_score: f64,
    pub score_delta: f64,
    pub changed_count: usize,
    pub total_distance: f64,
}

/// Correction display policy.
#[derive(Debug, Clone)]
pub struct CorrectionPolicy {
    pub config: CorrectionPolicyConfig,
}

impl CorrectionPolicy {
    pub fn new(config: CorrectionPolicyConfig) -> Self {
        Self { config }
    }

    pub fn max_allowed_changes(&self, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        // A misplaced space always changes two characters, so short sentences
        // must still allow two changes (PRD: short-sentence exception).
        let by_frac = ((len as f64) * self.config.max_changed_fraction).floor() as usize;
        by_frac
            .min(self.config.max_changed_absolute)
            .max(2.min(len))
    }

    /// Skip whole-sentence correction for English / URL / email / password / code / mostly-digits.
    pub fn should_skip_session(&self, session: &SentenceSession) -> bool {
        should_skip_text(&session.original_text)
    }

    pub fn select_suggestion(
        &self,
        session: &SentenceSession,
        ranked: &[RankedSentence],
        lm: &dyn LanguageModel,
    ) -> Option<CorrectionSuggestion> {
        if session.is_empty() || ranked.is_empty() {
            return None;
        }
        if self.should_skip_session(session) {
            return None;
        }

        let original_score = lm.score_sentence(&session.original_text);
        let max_changes = self.max_allowed_changes(session.len());

        for candidate in ranked {
            if candidate.text == session.original_text {
                continue;
            }
            if candidate.changed_count == 0 {
                continue;
            }
            if candidate.changed_count > max_changes {
                continue;
            }
            let delta = candidate.score - original_score;
            if delta < self.config.min_score_delta {
                continue;
            }
            return Some(CorrectionSuggestion {
                corrected_text: candidate.text.clone(),
                original_text: session.original_text.clone(),
                score: candidate.score,
                original_score,
                score_delta: delta,
                changed_count: candidate.changed_count,
                total_distance: candidate.total_distance,
            });
        }
        None
    }
}

fn should_skip_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    if looks_like_url(trimmed) || looks_like_email(trimmed) {
        return true;
    }
    if looks_like_password(trimmed) {
        return true;
    }
    if looks_like_code(trimmed) {
        return true;
    }
    if is_mostly_ascii_english(trimmed) {
        return true;
    }
    if is_mostly_digits(trimmed) {
        return true;
    }
    false
}

fn looks_like_url(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("www.")
        || lower.contains("://")
}

fn looks_like_email(text: &str) -> bool {
    let bytes = text.as_bytes();
    if let Some(at) = bytes.iter().position(|&b| b == b'@') {
        if at == 0 || at + 1 >= bytes.len() {
            return false;
        }
        return text[at + 1..].contains('.');
    }
    false
}

fn looks_like_password(text: &str) -> bool {
    // Heuristic: relatively long ASCII mix of letters+digits+symbols, no spaces/CJK.
    if text.chars().any(|c| c.is_whitespace() || !c.is_ascii()) {
        return false;
    }
    if text.len() < 8 {
        return false;
    }
    let has_letter = text.chars().any(|c| c.is_ascii_alphabetic());
    let has_digit = text.chars().any(|c| c.is_ascii_digit());
    let has_symbol = text.chars().any(|c| !c.is_ascii_alphanumeric());
    has_letter && has_digit && has_symbol
}

fn looks_like_code(text: &str) -> bool {
    if text.contains("```") || text.contains("=>") || text.contains("::") {
        return true;
    }
    if text.contains('{') && text.contains('}') {
        return true;
    }
    if text.contains(';') && text.contains('(') && text.contains(')') {
        return true;
    }
    false
}

fn is_mostly_ascii_english(text: &str) -> bool {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return false;
    }
    let ascii_alpha = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
    let cjk = chars
        .iter()
        .filter(|c| {
            let u = **c as u32;
            (0x4E00..=0x9FFF).contains(&u) || (0x3400..=0x4DBF).contains(&u)
        })
        .count();
    if cjk > 0 {
        return false;
    }
    (ascii_alpha as f64) / (chars.len() as f64) >= 0.8
}

fn is_mostly_digits(text: &str) -> bool {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return false;
    }
    let digits = chars.iter().filter(|c| c.is_ascii_digit()).count();
    (digits as f64) / (chars.len() as f64) >= 0.7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_url_email_english_digits() {
        assert!(should_skip_text("https://example.com/path"));
        assert!(should_skip_text("user@example.com"));
        assert!(should_skip_text("Hello world this is English"));
        assert!(should_skip_text("1234567890"));
        assert!(should_skip_text("fn main() { println!(\"x\"); }"));
        assert!(should_skip_text("P@ssw0rd!"));
        assert!(!should_skip_text("這樣如果偶爾"));
    }

    #[test]
    fn max_changes_rule() {
        let policy = CorrectionPolicy::new(CorrectionPolicyConfig::mvp_defaults());
        // len 16 → 25% = 4, min with 3 → 3
        assert_eq!(policy.max_allowed_changes(16), 3);
        // len 8 → 25% = 2
        assert_eq!(policy.max_allowed_changes(8), 2);
        assert_eq!(policy.max_allowed_changes(3), 2);
        assert_eq!(policy.max_allowed_changes(1), 1);
    }
}
