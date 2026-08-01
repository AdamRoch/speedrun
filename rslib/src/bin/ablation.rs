// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Runs the three-arm probe ablation experiment (simulated learners) and
//! writes the measured results as JSON. Two phases, orchestrated by
//! `just ablation`:
//!
//! 1. `ablation --emit-collection <path>` writes the synthetic corpus.
//! 2. `just probe-gen <path> --deck Default [--baseline]` attaches probes.
//! 3. `ablation --collection <path> --out <json>` runs the three arms.
//!
//! See `anki::ablation` and `data/ablation/report.md`.

use std::fs;
use std::path::PathBuf;

use anki::ablation::emit_corpus_collection;
use anki::ablation::run;
use anki::ablation::AblationConfig;
use anki::ablation::FALSIFICATION;

fn main() {
    let mut cfg = AblationConfig::default();
    let mut emit: Option<PathBuf> = None;
    let mut collection: Option<PathBuf> = None;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut val = |name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{name} needs a value"))
        };
        match arg.as_str() {
            "--seed" => cfg.seed = val("--seed").parse().unwrap(),
            "--learners" => cfg.learners = val("--learners").parse().unwrap(),
            "--cards" => cfg.cards = val("--cards").parse().unwrap(),
            "--days" => cfg.days = val("--days").parse().unwrap(),
            "--emit-collection" => emit = Some(val("--emit-collection").into()),
            "--collection" => collection = Some(val("--collection").into()),
            "--out" => out = Some(val("--out")),
            other => panic!(
                "unknown argument: {other} (expected --seed/--learners/--cards/--days/--emit-collection/--collection/--out)"
            ),
        }
    }

    if let Some(path) = emit {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("creating corpus dir");
        }
        emit_corpus_collection(&cfg, &path).expect("emitting corpus collection");
        eprintln!(
            "corpus written to {} ({} synthetic cards, seed {}); now attach probes with: just probe-gen {} --deck Default --baseline",
            path.display(),
            cfg.cards,
            cfg.seed,
            path.display(),
        );
        return;
    }

    let corpus = collection.expect("pass --emit-collection <path> or --collection <path>");
    eprintln!(
        "ablation: seed={} learners={}/arm days={} (simulated learners)",
        cfg.seed, cfg.learners, cfg.days
    );
    eprintln!("falsification condition (pre-stated): {FALSIFICATION}");
    let results = run(&cfg, &corpus).expect("ablation run failed");
    let pretty = serde_json::to_string_pretty(&results).unwrap();
    if let Some(path) = out {
        fs::write(&path, &pretty).expect("writing results");
        eprintln!("results written to {path}");
    }
    println!("{pretty}");
}
