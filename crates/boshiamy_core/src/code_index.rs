//! Bidirectional code ↔ character index.

use crate::cin_parser::CinTable;
use std::collections::{BTreeMap, BTreeSet};

/// Bidirectional index: code→chars and char→codes (multi-code per char preserved).
#[derive(Debug, Clone, Default)]
pub struct CodeIndex {
    code_to_chars: BTreeMap<String, BTreeSet<char>>,
    char_to_codes: BTreeMap<char, BTreeSet<String>>,
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
        self.char_to_codes.entry(ch).or_default().insert(code);
    }

    pub fn chars_for_code(&self, code: &str) -> impl Iterator<Item = char> + '_ {
        self.code_to_chars
            .get(code)
            .into_iter()
            .flatten()
            .copied()
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
        self.code_to_chars.iter().flat_map(|(code, chars)| {
            chars.iter().map(move |ch| (code.as_str(), *ch))
        })
    }
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
}
