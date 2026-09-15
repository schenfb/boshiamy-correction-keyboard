//! Unit coverage for parse, index, distance, candidates, ranking, policy.

use boshiamy_core::{
    BoshiamyDistance, CandidateGenerator, CinParser, CodeIndex, CorrectionPolicy,
    CorrectionPolicyConfig, LanguageModel, ScoringWeights, SentenceRanker, SentenceSession,
    SessionUnit, StubNgramModel,
};

const TINY_CIN: &str = r#"
%ename Tiny
%chardef begin
aa 測
ab 試
ba 如
bb 甘
ca 偶
cb 側
da 爾
db 而
%chardef end
"#;

#[test]
fn cin_parser_rejects_unclosed_and_empty() {
    assert!(CinParser::parse("").is_err());
    assert!(CinParser::parse("%chardef begin\naa 測\n").is_err());
    assert!(CinParser::parse("%ename X\n").is_err());
}

#[test]
fn code_index_multi_code_per_char() {
    let table = CinParser::parse(
        "%chardef begin\naa 字\nab 字\nac 詞\n%chardef end\n",
    )
    .unwrap();
    let index = CodeIndex::from_cin_table(&table);
    let mut codes: Vec<_> = index.codes_for_char('字').collect();
    codes.sort();
    assert_eq!(codes, vec!["aa", "ab"]);
}

#[test]
fn boshiamy_distance_mvp_substitution_only() {
    assert_eq!(BoshiamyDistance::substitution_distance("ca", "ca"), Some(0.0));
    assert_eq!(BoshiamyDistance::substitution_distance("ca", "cb"), Some(1.0));
    assert_eq!(BoshiamyDistance::substitution_distance("ca", "c"), None);
    assert_eq!(BoshiamyDistance::substitution_distance("ca", "cab"), None);
    assert_eq!(BoshiamyDistance::substitution_distance("ab", "ba"), None);
}

#[test]
fn candidate_generator_caps_and_keeps_original() {
    let index = CodeIndex::from_cin_table(&CinParser::parse(TINY_CIN).unwrap());
    let gen = CandidateGenerator::new(2);
    let session = SentenceSession::from_units(vec![SessionUnit::new('甘', "bb")]);
    let cands = gen.generate(&index, &session);
    assert!(cands[0].len() <= 2);
    assert_eq!(cands[0][0].character, '甘');
    assert!(cands[0][0].is_original);
}

#[test]
fn sentence_ranker_beam_prefers_better_lm_path() {
    let index = CodeIndex::from_cin_table(&CinParser::parse(TINY_CIN).unwrap());
    let session = SentenceSession::from_units(vec![
        SessionUnit::new('甘', "bb"),
        SessionUnit::new('側', "cb"),
        SessionUnit::new('而', "db"),
    ]);
    let gen = CandidateGenerator::new(8);
    let per = gen.generate(&index, &session);
    let lm = StubNgramModel::default_traditional_chinese_stub();
    let ranker = SentenceRanker::new(ScoringWeights::mvp_defaults());
    let ranked = ranker.rank(&session, &per, &lm);
    assert!(!ranked.is_empty());
    // Best should include 如/偶/爾 when LM boosts them.
    let texts: Vec<_> = ranked.iter().map(|r| r.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| *t == "如偶爾"),
        "expected 如偶爾 among beam results, got: {texts:?}"
    );
    let original = ranked.iter().find(|r| r.text == "甘側而").expect("original path");
    let corrected = ranked.iter().find(|r| r.text == "如偶爾").unwrap();
    assert!(
        corrected.score > original.score,
        "如偶爾 score {} should beat 甘側而 {}",
        corrected.score,
        original.score
    );
}

#[test]
fn correction_policy_requires_delta_and_change_cap() {
    let lm = StubNgramModel::default_traditional_chinese_stub();
    let policy = CorrectionPolicy::new(CorrectionPolicyConfig {
        min_score_delta: 1000.0, // impossible
        max_changed_absolute: 3,
        max_changed_fraction: 0.25,
    });
    let session = SentenceSession::from_units(vec![
        SessionUnit::new('甘', "bb"),
        SessionUnit::new('果', "bc"),
    ]);
    // Fabricate a ranked sentence that differs but cannot beat huge delta.
    let ranked = vec![boshiamy_core::RankedSentence {
        text: "如果".into(),
        chars: vec!['如', '果'],
        total_distance: 1.0,
        changed_count: 1,
        changed_explicit_selections: 0,
        language_score: lm.score_sentence("如果"),
        score: lm.score_sentence("如果"),
    }];
    assert!(policy.select_suggestion(&session, &ranked, &lm).is_none());
}

#[test]
fn policy_skips_english_and_urls() {
    let policy = CorrectionPolicy::new(CorrectionPolicyConfig::mvp_defaults());
    let eng = SentenceSession::from_units(
        "Hello"
            .chars()
            .map(|c| SessionUnit::new(c, "xx"))
            .collect(),
    );
    assert!(policy.should_skip_session(&eng));
}
