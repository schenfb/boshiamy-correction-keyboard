//! Offline character trigram language model loaded from the `BSLM` binary format
//! produced by `boshiamy_lm_train`.
//!
//! Format (little-endian):
//! ```text
//! magic "BSLM" | version u32 (=1) | order u32 (=3)
//! n_vocab u32 | codepoints u32[n_vocab]            (ids 3.. ; 0=<s> 1=</s> 2=<unk>)
//! unk_logprob f32
//! uni_logprob f32[n_vocab+3] | uni_backoff f32[n_vocab+3]
//! n_bigram u32 | keys u64[n] (a<<32|b) sorted | logprob f32[n] | backoff f32[n]
//! n_trigram u32 | keys u64[n] (a<<40|b<<20|c) sorted | logprob f32[n]
//! ```
//! All log-probabilities are natural logs.

use crate::language_model::LanguageModel;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read};

pub const BOS: u32 = 0;
pub const EOS: u32 = 1;
pub const UNK: u32 = 2;
pub const FIRST_CHAR_ID: u32 = 3;

#[derive(Debug)]
pub enum NgramLoadError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    UnsupportedOrder(u32),
    Truncated,
}

impl fmt::Display for NgramLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NgramLoadError::Io(e) => write!(f, "io error: {e}"),
            NgramLoadError::BadMagic => write!(f, "not a BSLM model file"),
            NgramLoadError::UnsupportedVersion(v) => write!(f, "unsupported BSLM version {v}"),
            NgramLoadError::UnsupportedOrder(o) => write!(f, "unsupported n-gram order {o}"),
            NgramLoadError::Truncated => write!(f, "model file truncated"),
        }
    }
}

impl std::error::Error for NgramLoadError {}

impl From<io::Error> for NgramLoadError {
    fn from(e: io::Error) -> Self {
        NgramLoadError::Io(e)
    }
}

/// Character trigram model with Katz/KN-style backoff weights.
pub struct NgramModel {
    char_to_id: HashMap<char, u32>,
    unk_logprob: f32,
    uni_logprob: Vec<f32>,
    uni_backoff: Vec<f32>,
    bi_keys: Vec<u64>,
    bi_logprob: Vec<f32>,
    bi_backoff: Vec<f32>,
    tri_keys: Vec<u64>,
    tri_logprob: Vec<f32>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], NgramLoadError> {
        if self.pos + n > self.buf.len() {
            return Err(NgramLoadError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, NgramLoadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, NgramLoadError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, NgramLoadError> {
        Ok(self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, NgramLoadError> {
        Ok(self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
    fn u64s(&mut self, n: usize) -> Result<Vec<u64>, NgramLoadError> {
        Ok(self
            .take(n * 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect())
    }
}

#[inline]
fn bi_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

#[inline]
fn tri_key(a: u32, b: u32, c: u32) -> u64 {
    ((a as u64) << 40) | ((b as u64) << 20) | c as u64
}

impl NgramModel {
    pub fn from_bytes(buf: &[u8]) -> Result<Self, NgramLoadError> {
        let mut c = Cursor { buf, pos: 0 };
        if c.take(4)? != b"BSLM" {
            return Err(NgramLoadError::BadMagic);
        }
        let version = c.u32()?;
        if version != 1 {
            return Err(NgramLoadError::UnsupportedVersion(version));
        }
        let order = c.u32()?;
        if order != 3 {
            return Err(NgramLoadError::UnsupportedOrder(order));
        }
        let n_vocab = c.u32()? as usize;
        let cps = c.u32s(n_vocab)?;
        let mut char_to_id = HashMap::with_capacity(n_vocab);
        for (i, cp) in cps.iter().enumerate() {
            if let Some(ch) = char::from_u32(*cp) {
                char_to_id.insert(ch, FIRST_CHAR_ID + i as u32);
            }
        }
        let unk_logprob = c.f32()?;
        let total = n_vocab + FIRST_CHAR_ID as usize;
        let uni_logprob = c.f32s(total)?;
        let uni_backoff = c.f32s(total)?;
        let n_bi = c.u32()? as usize;
        let bi_keys = c.u64s(n_bi)?;
        let bi_logprob = c.f32s(n_bi)?;
        let bi_backoff = c.f32s(n_bi)?;
        let n_tri = c.u32()? as usize;
        let tri_keys = c.u64s(n_tri)?;
        let tri_logprob = c.f32s(n_tri)?;
        Ok(Self {
            char_to_id,
            unk_logprob,
            uni_logprob,
            uni_backoff,
            bi_keys,
            bi_logprob,
            bi_backoff,
            tri_keys,
            tri_logprob,
        })
    }

    pub fn from_reader<R: Read>(mut r: R) -> Result<Self, NgramLoadError> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        Self::from_bytes(&buf)
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, NgramLoadError> {
        let buf = std::fs::read(path)?;
        Self::from_bytes(&buf)
    }

    pub fn vocab_size(&self) -> usize {
        self.char_to_id.len()
    }
    pub fn bigram_count(&self) -> usize {
        self.bi_keys.len()
    }
    pub fn trigram_count(&self) -> usize {
        self.tri_keys.len()
    }

    #[inline]
    pub fn id(&self, ch: char) -> u32 {
        self.char_to_id.get(&ch).copied().unwrap_or(UNK)
    }

    #[inline]
    fn bi(&self, a: u32, b: u32) -> Option<usize> {
        self.bi_keys.binary_search(&bi_key(a, b)).ok()
    }

    #[inline]
    fn tri(&self, a: u32, b: u32, c: u32) -> Option<usize> {
        self.tri_keys.binary_search(&tri_key(a, b, c)).ok()
    }

    /// log P(c | a b) with backoff.
    pub fn logprob(&self, a: u32, b: u32, c: u32) -> f32 {
        if let Some(i) = self.tri(a, b, c) {
            return self.tri_logprob[i];
        }
        let bo_ab = self.bi(a, b).map(|i| self.bi_backoff[i]).unwrap_or(0.0);
        if let Some(i) = self.bi(b, c) {
            return bo_ab + self.bi_logprob[i];
        }
        let bo_b = self.uni_backoff[b as usize];
        let uni = if c == UNK {
            self.unk_logprob
        } else {
            self.uni_logprob[c as usize]
        };
        bo_ab + bo_b + uni
    }

    /// Score a full sentence including `<s> <s>` start and `</s>` end.
    pub fn score_ids(&self, ids: &[u32]) -> f32 {
        let mut a = BOS;
        let mut b = BOS;
        let mut total = 0.0f32;
        for &c in ids {
            total += self.logprob(a, b, c);
            a = b;
            b = c;
        }
        total + self.logprob(a, b, EOS)
    }
}

impl LanguageModel for NgramModel {
    fn unigram(&self, ch: char) -> f64 {
        let c = self.id(ch);
        if c == UNK {
            self.unk_logprob as f64
        } else {
            self.uni_logprob[c as usize] as f64
        }
    }

    fn score_sentence(&self, text: &str) -> f64 {
        let ids: Vec<u32> = text.chars().map(|c| self.id(c)).collect();
        self.score_ids(&ids) as f64
    }

    fn score_transition(&self, prefix: &str, next: char) -> f64 {
        let mut it = prefix.chars().rev();
        let b = it.next().map(|c| self.id(c)).unwrap_or(BOS);
        let a = it.next().map(|c| self.id(c)).unwrap_or(BOS);
        self.logprob(a, b, self.id(next)) as f64
    }
}
