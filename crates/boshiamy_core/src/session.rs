//! Sentence session model for tracking typed units.

/// One committed character and the raw key codes that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionUnit {
    pub output_character: char,
    /// Raw Boshiamy-style letter codes as typed (ASCII letters, typically lowercase).
    pub raw_code: String,
    /// Candidate list index chosen by the user; `0` or negative means default/first.
    /// Values `> 0` mean the user explicitly picked a non-default candidate.
    pub selected_index: i32,
    /// Opaque timestamp for session validity (not used in scoring MVP).
    pub timestamp: u64,
}

impl SessionUnit {
    pub fn new(output_character: char, raw_code: impl Into<String>) -> Self {
        Self {
            output_character,
            raw_code: raw_code.into(),
            selected_index: 0,
            timestamp: 0,
        }
    }

    pub fn with_selection(mut self, selected_index: i32) -> Self {
        self.selected_index = selected_index;
        self
    }

    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// True when the user actively chose a non-first candidate.
    pub fn is_explicit_selection(&self) -> bool {
        self.selected_index > 0
    }
}

/// In-memory session for the current sentence being typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceSession {
    pub original_text: String,
    pub units: Vec<SessionUnit>,
}

impl SentenceSession {
    pub fn from_units(units: Vec<SessionUnit>) -> Self {
        let original_text: String = units.iter().map(|u| u.output_character).collect();
        Self {
            original_text,
            units,
        }
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }
}
