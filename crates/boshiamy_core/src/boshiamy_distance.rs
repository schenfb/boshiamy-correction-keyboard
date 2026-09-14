//! Boshiamy-style key-code distance (MVP: substitution only).

/// Distance between typed raw codes and a candidate's legal codes.
///
/// MVP rules:
/// - identical codes → `0.0`
/// - same length, exactly one letter substituted → `1.0`
/// - insert / delete / transpose → not implemented (returns `None`)
pub struct BoshiamyDistance;

impl BoshiamyDistance {
    /// Distance when both strings are already known; `None` if unsupported/beyond MVP.
    pub fn substitution_distance(a: &str, b: &str) -> Option<f64> {
        if a == b {
            return Some(0.0);
        }
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        if a_bytes.len() != b_bytes.len() {
            return None;
        }
        let mut diffs = 0usize;
        for (x, y) in a_bytes.iter().zip(b_bytes.iter()) {
            if x != y {
                diffs += 1;
                if diffs > 1 {
                    return None;
                }
            }
        }
        if diffs == 1 {
            Some(1.0)
        } else {
            None
        }
    }

    /// Minimum distance from `raw_code` to any of `legal_codes`.
    /// Returns `Some(0.0)` / `Some(1.0)` or `None` if no MVP-supported neighbor.
    pub fn min_distance_to_codes<'a, I>(raw_code: &str, legal_codes: I) -> Option<f64>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut best: Option<f64> = None;
        for code in legal_codes {
            if let Some(d) = Self::substitution_distance(raw_code, code) {
                best = Some(match best {
                    Some(b) => b.min(d),
                    None => d,
                });
                if best == Some(0.0) {
                    break;
                }
            }
        }
        best
    }

    /// Distance from typed `raw_code` to character `ch` given an index of legal codes.
    pub fn distance_to_char<'a, I>(raw_code: &str, legal_codes: I) -> Option<f64>
    where
        I: IntoIterator<Item = &'a str>,
    {
        Self::min_distance_to_codes(raw_code, legal_codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_zero() {
        assert_eq!(BoshiamyDistance::substitution_distance("ba", "ba"), Some(0.0));
    }

    #[test]
    fn one_letter_replace_is_one() {
        assert_eq!(BoshiamyDistance::substitution_distance("ba", "bb"), Some(1.0));
        assert_eq!(BoshiamyDistance::substitution_distance("ca", "cb"), Some(1.0));
    }

    #[test]
    fn insert_delete_transpose_not_supported() {
        assert_eq!(BoshiamyDistance::substitution_distance("ba", "b"), None);
        assert_eq!(BoshiamyDistance::substitution_distance("ba", "baa"), None);
        // two substitutions
        assert_eq!(BoshiamyDistance::substitution_distance("ba", "cd"), None);
        // transpose of adjacent would be same length with two diffs → unsupported in MVP
        assert_eq!(BoshiamyDistance::substitution_distance("ab", "ba"), None);
    }
}
