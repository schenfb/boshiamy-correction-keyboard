//! UTF-8 `.cin` table parser (OpenVanilla / libcangjie-style subset).

use std::collections::BTreeMap;
use std::fmt;

/// Parsed `.cin` content before indexing.
#[derive(Debug, Clone, Default)]
pub struct CinTable {
    pub name: Option<String>,
    pub ename: Option<String>,
    /// Ordered (code, character) mappings from `%chardef`.
    pub mappings: Vec<(String, char)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CinParseError {
    EmptyInput,
    InvalidLine { line: usize, content: String },
    MissingChardef,
    ChardefNotClosed,
}

impl fmt::Display for CinParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CinParseError::EmptyInput => write!(f, "empty .cin input"),
            CinParseError::InvalidLine { line, content } => {
                write!(f, "invalid .cin line {line}: {content}")
            }
            CinParseError::MissingChardef => write!(f, "missing %chardef begin"),
            CinParseError::ChardefNotClosed => write!(f, "%chardef begin without end"),
        }
    }
}

impl std::error::Error for CinParseError {}

/// Parser for a minimal UTF-8 `.cin` dialect.
pub struct CinParser;

impl CinParser {
    /// Parse UTF-8 `.cin` text into a [`CinTable`].
    ///
    /// Supported:
    /// - Header keys `%ename`, `%cname`, `%name` (stored when present)
    /// - `%chardef begin` … `%chardef end` with lines `code<TAB/SPACE>character`
    /// - Comments starting with `#`
    /// - Multi-code per character (repeated character with different codes)
    pub fn parse(text: &str) -> Result<CinTable, CinParseError> {
        if text.trim().is_empty() {
            return Err(CinParseError::EmptyInput);
        }

        let mut table = CinTable::default();
        let mut in_chardef = false;
        let mut saw_chardef = false;
        let mut properties: BTreeMap<String, String> = BTreeMap::new();

        for (idx, raw_line) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.eq_ignore_ascii_case("%chardef begin") {
                in_chardef = true;
                saw_chardef = true;
                continue;
            }
            if line.eq_ignore_ascii_case("%chardef end") {
                in_chardef = false;
                continue;
            }

            if !in_chardef {
                if let Some(rest) = line.strip_prefix('%') {
                    let mut parts = rest.splitn(2, char::is_whitespace);
                    if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                        properties.insert(key.to_ascii_lowercase(), value.trim().to_string());
                    }
                    continue;
                }
                // Ignore unknown preamble lines.
                continue;
            }

            let mut parts = line.split_whitespace();
            let code = parts.next().ok_or_else(|| CinParseError::InvalidLine {
                line: line_no,
                content: line.to_string(),
            })?;
            let ch_str = parts.next().ok_or_else(|| CinParseError::InvalidLine {
                line: line_no,
                content: line.to_string(),
            })?;
            let mut chars = ch_str.chars();
            let ch = chars.next().ok_or_else(|| CinParseError::InvalidLine {
                line: line_no,
                content: line.to_string(),
            })?;
            // Accept single Unicode scalar; ignore leftover combining junk by requiring exactly one.
            if chars.next().is_some() {
                return Err(CinParseError::InvalidLine {
                    line: line_no,
                    content: line.to_string(),
                });
            }
            // Real tables use letters plus a few punctuation keys (e.g. `,` prefixes
            // symbol codes in Boshiamy-style tables); accept any printable ASCII.
            let normalized = code.to_ascii_lowercase();
            if !normalized.chars().all(|c| c.is_ascii_graphic()) {
                return Err(CinParseError::InvalidLine {
                    line: line_no,
                    content: line.to_string(),
                });
            }
            table.mappings.push((normalized, ch));
        }

        if !saw_chardef {
            return Err(CinParseError::MissingChardef);
        }
        if in_chardef {
            return Err(CinParseError::ChardefNotClosed);
        }

        table.ename = properties.get("ename").cloned();
        table.name = properties
            .get("cname")
            .or_else(|| properties.get("name"))
            .cloned();

        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_cin() {
        let text = r#"
%ename Synthetic
%cname 合成碼表
%chardef begin
aa 這
ab 樣
ba 如
%chardef end
"#;
        let table = CinParser::parse(text).unwrap();
        assert_eq!(table.ename.as_deref(), Some("Synthetic"));
        assert_eq!(table.name.as_deref(), Some("合成碼表"));
        assert_eq!(table.mappings.len(), 3);
        assert_eq!(table.mappings[0], ("aa".into(), '這'));
    }
}
