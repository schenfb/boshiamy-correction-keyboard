//! AC2 acceptance: typed 「這樣甘果側而打錯一個字也沒關係」 corrects to
//! 「這樣如果偶爾打錯一個字也沒關係」 when wrong chars' rawCodes are distance 1
//! from the intended codes.
//!
//! Synthetic mapping (see fixtures/synthetic_ac2.cin):
//!   如 ba ↔ 甘 bb (typed bb)
//!   偶 ca ↔ 側 cb (typed cb)
//!   爾 da ↔ 而 db (typed db)

use boshiamy_core::{
    CinParser, CodeIndex, CorrectionEngine, SentenceSession, SessionUnit,
};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic_ac2.cin")
}

fn ac2_session() -> SentenceSession {
    // 這樣甘果側而打錯一個字也沒關係
    let pairs = [
        ('這', "aa"),
        ('樣', "ab"),
        ('甘', "bb"), // intended 如/ba
        ('果', "bc"),
        ('側', "cb"), // intended 偶/ca
        ('而', "db"), // intended 爾/da
        ('打', "ea"),
        ('錯', "eb"),
        ('一', "ec"),
        ('個', "ed"),
        ('字', "ee"),
        ('也', "ef"),
        ('沒', "eg"),
        ('關', "eh"),
        ('係', "ei"),
    ];
    SentenceSession::from_units(
        pairs
            .into_iter()
            .map(|(ch, code)| SessionUnit::new(ch, code))
            .collect(),
    )
}

#[test]
fn ac2_corrects_three_distance_one_typos() {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture");
    let table = CinParser::parse(&text).expect("parse");
    let index = CodeIndex::from_cin_table(&table);
    let engine = CorrectionEngine::with_defaults(index);

    let session = ac2_session();
    assert_eq!(session.original_text, "這樣甘果側而打錯一個字也沒關係");

    let suggestion = engine
        .suggest(&session)
        .expect("expected a correction suggestion");
    assert_eq!(suggestion.corrected_text, "這樣如果偶爾打錯一個字也沒關係");
    assert_eq!(suggestion.changed_count, 3);
    assert!(suggestion.score_delta > 0.0);
}

#[test]
fn parse_index_distance_candidates_smoke() {
    use boshiamy_core::{BoshiamyDistance, CandidateGenerator};

    let text = std::fs::read_to_string(fixture_path()).unwrap();
    let table = CinParser::parse(&text).unwrap();
    assert!(table.mappings.len() >= 15);
    let index = CodeIndex::from_cin_table(&table);
    assert!(index.codes_for_char('如').any(|c| c == "ba"));
    assert_eq!(
        BoshiamyDistance::substitution_distance("bb", "ba"),
        Some(1.0)
    );

    let session = ac2_session();
    let gen = CandidateGenerator::new(8);
    let cands = gen.generate(&index, &session);
    let pos2: Vec<char> = cands[2].iter().map(|c| c.character).collect();
    assert!(pos2.contains(&'甘'));
    assert!(pos2.contains(&'如'));
}
