//! CLI: load a synthetic/user `.cin`, score a sentence session, print correction or none.
//!
//! Example (AC2):
//!   boshiamy-correct \
//!     --cin crates/boshiamy_core/tests/fixtures/synthetic_ac2.cin \
//!     --units "這:aa,樣:ab,甘:bb,果:bc,側:cb,而:db,打:ea,錯:eb,一:ec,個:ed,字:ee,也:ef,沒:eg,關:eh,係:ei"

use boshiamy_core::{CinParser, CodeIndex, CorrectionEngine, SentenceSession, SessionUnit};
use std::env;
use std::fs;
use std::process::ExitCode;

fn print_usage() {
    eprintln!(
        "Usage:\n  boshiamy-correct --cin <path.cin> --units \"字:code,字:code,...\"\n\n\
         Units format: outputCharacter:rawCode pairs separated by commas.\n\
         Prints the best correction sentence, or the line \"none\"."
    );
}

fn parse_units(spec: &str) -> Result<Vec<SessionUnit>, String> {
    let mut units = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (ch_str, code) = part
            .split_once(':')
            .ok_or_else(|| format!("invalid unit (expected CHAR:code): {part}"))?;
        let mut chars = ch_str.chars();
        let ch = chars
            .next()
            .ok_or_else(|| format!("missing character in unit: {part}"))?;
        if chars.next().is_some() {
            return Err(format!("unit character must be a single char: {part}"));
        }
        units.push(SessionUnit::new(ch, code.trim().to_ascii_lowercase()));
    }
    if units.is_empty() {
        return Err("no units provided".into());
    }
    Ok(units)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut cin_path: Option<String> = None;
    let mut units_spec: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cin" => {
                i += 1;
                cin_path = args.get(i).cloned();
            }
            "--units" => {
                i += 1;
                units_spec = args.get(i).cloned();
            }
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_usage();
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let (Some(cin_path), Some(units_spec)) = (cin_path, units_spec) else {
        print_usage();
        return ExitCode::from(2);
    };

    let cin_text = match fs::read_to_string(&cin_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to read {cin_path}: {e}");
            return ExitCode::from(1);
        }
    };

    let table = match CinParser::parse(&cin_text) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("CIN parse error: {e}");
            return ExitCode::from(1);
        }
    };

    let units = match parse_units(&units_spec) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    let index = CodeIndex::from_cin_table(&table);
    let engine = CorrectionEngine::with_defaults(index);
    let session = SentenceSession::from_units(units);

    match engine.suggest(&session) {
        Some(s) => {
            println!("{}", s.corrected_text);
            ExitCode::SUCCESS
        }
        None => {
            println!("none");
            ExitCode::SUCCESS
        }
    }
}
