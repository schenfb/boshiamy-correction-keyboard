//! Per-position candidate generation (original + distance ≤ 1).

use crate::boshiamy_distance::BoshiamyDistance;
use crate::code_index::CodeIndex;
use crate::session::SentenceSession;
use std::collections::BTreeMap;

/// A candidate character at one sentence position.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionCandidate {
    pub character: char,
    pub distance: f64,
    /// True when this is the originally committed character.
    pub is_original: bool,
}

/// Generates per-position alternatives capped to avoid combinatorial explosion.
#[derive(Debug, Clone)]
pub struct CandidateGenerator {
    pub max_per_position: usize,
}

impl CandidateGenerator {
    pub fn new(max_per_position: usize) -> Self {
        Self {
            max_per_position: max_per_position.max(1),
        }
    }

    /// For each session unit, keep the original character plus distance≤1 alternatives.
    pub fn generate(
        &self,
        index: &CodeIndex,
        session: &SentenceSession,
    ) -> Vec<Vec<PositionCandidate>> {
        session
            .units
            .iter()
            .map(|unit| self.candidates_for_unit(index, unit.output_character, &unit.raw_code))
            .collect()
    }

    fn candidates_for_unit(
        &self,
        index: &CodeIndex,
        original: char,
        raw_code: &str,
    ) -> Vec<PositionCandidate> {
        let mut by_char: BTreeMap<char, PositionCandidate> = BTreeMap::new();

        // Always keep the original character at distance 0 (baseline "keep typed").
        by_char.insert(
            original,
            PositionCandidate {
                character: original,
                distance: 0.0,
                is_original: true,
            },
        );

        // Scan all codes at MVP substitution distance 0 or 1 from raw_code.
        for (code, ch) in index.iter_mappings() {
            let Some(dist) = BoshiamyDistance::substitution_distance(raw_code, code) else {
                continue;
            };
            if dist > 1.0 {
                continue;
            }
            by_char
                .entry(ch)
                .and_modify(|existing| {
                    if dist < existing.distance {
                        existing.distance = dist;
                    }
                })
                .or_insert(PositionCandidate {
                    character: ch,
                    distance: dist,
                    is_original: ch == original,
                });
        }

        let mut list: Vec<PositionCandidate> = by_char.into_values().collect();
        // Prefer original first, then lower distance, then stable by char.
        list.sort_by(|a, b| {
            b.is_original
                .cmp(&a.is_original)
                .then(a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.character.cmp(&b.character))
        });
        list.truncate(self.max_per_position);
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cin_parser::CinParser;
    use crate::session::SessionUnit;

    fn tiny_index() -> CodeIndex {
        let table = CinParser::parse(
            r#"
%chardef begin
ba 如
bb 甘
ca 偶
cb 側
%chardef end
"#,
        )
        .unwrap();
        CodeIndex::from_cin_table(&table)
    }

    #[test]
    fn keeps_original_and_distance_one_alts() {
        let index = tiny_index();
        let gen = CandidateGenerator::new(8);
        let session = SentenceSession::from_units(vec![SessionUnit::new('甘', "bb")]);
        let cands = gen.generate(&index, &session);
        assert_eq!(cands.len(), 1);
        let chars: Vec<char> = cands[0].iter().map(|c| c.character).collect();
        assert!(chars.contains(&'甘'));
        assert!(chars.contains(&'如')); // ba vs bb
        let like = cands[0].iter().find(|c| c.character == '如').unwrap();
        assert_eq!(like.distance, 1.0);
    }
}
