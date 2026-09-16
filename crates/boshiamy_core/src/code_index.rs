//! Bidirectional code ↔ character index.

use crate::cin_parser::CinTable;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Bidirectional index: code→chars and char→codes (multi-code per char preserved).
#[derive(Debug, Clone, Default)]
pub struct CodeIndex {
    code_to_chars: BTreeMap<String, BTreeSet<char>>,
    char_to_codes: BTreeMap<char, BTreeSet<String>>,
    /// Wildcard patterns (`a?c`) → chars whose code matches with one substitution.
    /// Built once so substitution-distance-1 lookups are O(code length), not O(table).
    wildcard_to_chars: HashMap<String, Vec<(String, char)>>,
    /// Code with one letter removed → entries, for "user skipped a letter" lookups.
    deletion_to_chars: HashMap<String, Vec<(String, char)>>,
}

/// One edit away from the typed code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditKind {
    /// One letter replaced.
    Substitution,
    /// Typed code is missing one letter of the legal code.
    Omission,
    /// Typed code has one extra letter.
    Insertion,
    /// Two adjacent letters swapped.
    Transposition,
}

impl CodeIndex {
    pub fn from_cin_table(table: &CinTable) -> Self {
        let mut index = Self::default();
        for (code, ch) in &table.mappings {
            index.insert(code.clone(), *ch);
        }
        index
    }

    pub fn insert(&mut self, code: String, ch: char) {
        self.code_to_chars
            .entry(code.clone())
            .or_default()
            .insert(ch);
        self.char_to_codes
            .entry(ch)
            .or_default()
            .insert(code.clone());
        for pat in wildcard_patterns(&code) {
            self.wildcard_to_chars
                .entry(pat)
                .or_default()
                .push((code.clone(), ch));
        }
        for shorter in deletion_patterns(&code) {
            self.deletion_to_chars
                .entry(shorter)
                .or_default()
                .push((code.clone(), ch));
        }
    }

    /// All (code, char, edit) triples exactly one edit away from `raw_code`
    /// (substitution, omission, insertion, adjacent transposition). Exact matches excluded.
    pub fn edit_neighbors(&self, raw_code: &str) -> Vec<(&str, char, EditKind)> {
        let mut out: Vec<(&str, char, EditKind)> = self
            .substitution_neighbors(raw_code)
            .into_iter()
            .map(|(c, ch)| (c, ch, EditKind::Substitution))
            .collect();
        if let Some(list) = self.deletion_to_chars.get(raw_code) {
            for (code, ch) in list {
                out.push((code.as_str(), *ch, EditKind::Omission));
            }
        }
        for shorter in deletion_patterns(raw_code) {
            if let Some((code, _)) = self.code_to_chars.get_key_value(&shorter) {
                for ch in self.chars_for_code(&shorter) {
                    out.push((code.as_str(), ch, EditKind::Insertion));
                }
            }
        }
        let bytes = raw_code.as_bytes();
        for i in 0..bytes.len().saturating_sub(1) {
            if bytes[i] == bytes[i + 1] {
                continue;
            }
            let mut swapped = bytes.to_vec();
            swapped.swap(i, i + 1);
            let swapped = String::from_utf8(swapped).unwrap_or_default();
            if let Some((code, _)) = self.code_to_chars.get_key_value(&swapped) {
                for ch in self.chars_for_code(&swapped) {
                    out.push((code.as_str(), ch, EditKind::Transposition));
                }
            }
        }
        out
    }

    /// All (code, char) pairs whose code is exactly one substitution away from `raw_code`.
    /// Exact matches are excluded; use [`chars_for_code`](Self::chars_for_code) for distance 0.
    pub fn substitution_neighbors(&self, raw_code: &str) -> Vec<(&str, char)> {
        let mut out = Vec::new();
        for pat in wildcard_patterns(raw_code) {
            if let Some(list) = self.wildcard_to_chars.get(&pat) {
                for (code, ch) in list {
                    if code != raw_code {
                        out.push((code.as_str(), *ch));
                    }
                }
            }
        }
        out
    }

    pub fn chars_for_code(&self, code: &str) -> impl Iterator<Item = char> + '_ {
        self.code_to_chars.get(code).into_iter().flatten().copied()
    }

    pub fn codes_for_char(&self, ch: char) -> impl Iterator<Item = &str> + '_ {
        self.char_to_codes
            .get(&ch)
            .into_iter()
            .flatten()
            .map(|s| s.as_str())
    }

    pub fn contains_code(&self, code: &str) -> bool {
        self.code_to_chars.contains_key(code)
    }

    pub fn all_codes(&self) -> impl Iterator<Item = &str> + '_ {
        self.code_to_chars.keys().map(|s| s.as_str())
    }

    pub fn code_count(&self) -> usize {
        self.code_to_chars.len()
    }

    pub fn char_count(&self) -> usize {
        self.char_to_codes.len()
    }

    /// Return every (code, char) pair for neighbor search helpers.
    pub fn iter_mappings(&self) -> impl Iterator<Item = (&str, char)> + '_ {
        self.code_to_chars
            .iter()
            .flat_map(|(code, chars)| chars.iter().map(move |ch| (code.as_str(), *ch)))
    }
}

/// `abc` → [`bc`, `ac`, `ab`] (only for codes of length ≥ 2).
fn deletion_patterns(code: &str) -> Vec<String> {
    let bytes = code.as_bytes();
    if bytes.len() < 2 {
        return Vec::new();
    }
    (0..bytes.len())
        .map(|i| {
            let mut p = String::with_capacity(bytes.len() - 1);
            for (j, &b) in bytes.iter().enumerate() {
                if i != j {
                    p.push(b as char);
                }
            }
            p
        })
        .collect()
}

/// `abc` → [`?bc`, `a?c`, `ab?`]
fn wildcard_patterns(code: &str) -> Vec<String> {
    let bytes = code.as_bytes();
    (0..bytes.len())
        .map(|i| {
            let mut p = String::with_capacity(bytes.len());
            for (j, &b) in bytes.iter().enumerate() {
                p.push(if i == j { '?' } else { b as char });
            }
            p
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cin_parser::CinParser;

    #[test]
    fn bidirectional_and_multi_code() {
        let table = CinParser::parse(
            r#"
%chardef begin
aa 一
ab 一
ac 二
%chardef end
"#,
        )
        .unwrap();
        let index = CodeIndex::from_cin_table(&table);
        let codes: BTreeSet<_> = index.codes_for_char('一').collect();
        assert_eq!(codes, BTreeSet::from(["aa", "ab"]));
        let chars: BTreeSet<_> = index.chars_for_code("aa").collect();
        assert_eq!(chars, BTreeSet::from(['一']));
    }

    #[test]
    fn substitution_neighbors_excludes_exact() {
        let table = CinParser::parse("%chardef begin\nba 如\nbb 甘\nab 樣\nbba 多\n%chardef end\n")
            .unwrap();
        let index = CodeIndex::from_cin_table(&table);
        let mut n: Vec<_> = index
            .substitution_neighbors("bb")
            .into_iter()
            .map(|(_, c)| c)
            .collect();
        n.sort();
        assert_eq!(n, vec!['如', '樣']);
    }
}
