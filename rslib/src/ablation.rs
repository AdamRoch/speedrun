// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Ascent fork: the three-arm probe ablation experiment.
//!
//! Tests the product's central claim - *a tool that measures only whether you
//! remember the card cannot tell you whether you know the thing* - against
//! **simulated learners** driven through the real `Collection`, the real FSRS
//! scheduler, and the real probe-substitution branch in
//! [crate::scheduler::queue].
//!
//! Arms:
//! - **stock**: no probes in the collection; the serving path short-circuits on
//!   the empty probes table, which is upstream Anki behaviour.
//! - **off**: probes attached, `probe_rate = 0`. The fork with the feature
//!   present but disabled. `zero_rate_never_substitutes` and the run-level
//!   equivalence check below guard that this arm schedules identically to
//!   stock.
//! - **on**: probes attached and served at the configured rate.
//!
//! The experiment runs in two phases so probes come from the real generation
//! pipeline (`pylib/anki/probe_gen.py`) rather than from harness-invented
//! text:
//!
//! 1. [emit_corpus_collection] writes a collection of synthetic fact cards.
//! 2. `just probe-gen <collection> --deck Default [--baseline]` attaches probes
//!    through the shipped pipeline and its quality gate.
//! 3. [run] copies that collection per (arm, learner) and simulates.
//!
//! Everything harness-side derives from one seed; with the `--baseline`
//! (no-AI, deterministic) generator the same seed reproduces the same
//! numbers bit for bit (`tiny_run_is_deterministic` guards this). AI-written
//! probes go through the identical path but are not reproducible; results
//! from them must be labelled as such. Learners are simulated; no human data
//! is involved anywhere. Probe *text* cannot influence a simulated learner's
//! outcomes - only the serving decision and the text-level leakage checks
//! see it.
//!
//! Falsification conditions are pre-stated in [FALSIFICATION] and evaluated
//! mechanically; the harness does not know which answer is flattering.

use std::collections::HashSet;
use std::path::Path;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde_json::json;
use serde_json::Value;

use crate::collection::CollectionBuilder;
use crate::config::BoolKey;
use crate::prelude::*;
use crate::scheduler::answering::CardAnswer;
use crate::scheduler::answering::Rating;
use crate::scheduler::queue::QueueEntryKind;

/// The claim is falsified if, on held-back cards:
/// (a) the high-retrievability gap `pass(original) - pass(probe)` is under
///     0.05 or its 95% CI includes zero - the probe adds no information; or
/// (b) FSRS retrievability predicts probe outcomes at least as well as it
///     predicts card outcomes (Brier(probe) <= Brier(card)) - remembering
///     the card already tells you whether you know the thing.
/// The run is invalid (regardless of outcome) if the off and stock arms do
/// not produce identical review sequences.
pub const FALSIFICATION: &str =
    "gap<0.05 or gap CI includes 0, or Brier(R->probe) <= Brier(R->card), on held-back cards";

const FIRST_CARD_ID: i64 = 100_000;

#[derive(Debug, Clone)]
pub struct AblationConfig {
    pub seed: u64,
    pub learners: usize,
    pub days: u32,
    /// Cards in the emitted corpus (phase 1 only; phase 2 uses whatever the
    /// collection contains).
    pub cards: usize,
    pub probe_rate: f32,
    pub probe_threshold: f32,
    /// Fraction of (learner, card) pairs learned by rote: high surface
    /// recall, low transfer to a reworded probe.
    pub rote_fraction: f64,
    /// Every n-th card is held back: its outcomes are used only for the
    /// reported metrics, never for tuning the harness.
    pub heldback_every: usize,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            learners: 8,
            cards: 200,
            days: 150,
            probe_rate: 0.5,
            probe_threshold: 0.85,
            rote_fraction: 0.4,
            heldback_every: 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Stock,
    Off,
    On,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Stock => "stock",
            Arm::Off => "off",
            Arm::On => "on",
        }
    }
}

/// Deterministic hash mixer (splitmix64) so every random stream is a pure
/// function of (seed, namespace...) and never of call order.
fn mix(parts: &[u64]) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for &p in parts {
        let mut z = h ^ p.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h = z ^ (z >> 31);
    }
    h
}

fn rng_for(seed: u64, ns: &str, parts: &[u64]) -> StdRng {
    let mut all = vec![seed, ns.bytes().fold(0u64, |a, b| a * 31 + b as u64)];
    all.extend_from_slice(parts);
    StdRng::seed_from_u64(mix(&all))
}

/// `SqliteStorage::get_all_cards` is test-only; this is the same thing for
/// the harness, via the public row shape.
fn all_cards(col: &Collection) -> Result<Vec<Card>> {
    let ids: Vec<CardId> = col
        .storage
        .db
        .prepare("select id from cards order by id")?
        .query_map([], |r| r.get(0).map(CardId))?
        .collect::<std::result::Result<_, _>>()?;
    ids.into_iter()
        .map(|id| col.storage.get_card(id)?.or_not_found(id))
        .collect()
}

fn pseudoword(rng: &mut StdRng, syllables: usize) -> String {
    const ONSET: &[&str] = &[
        "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "st", "tr",
    ];
    const NUCLEUS: &[&str] = &["a", "e", "i", "o", "u", "ai", "or", "an", "el"];
    let mut w = String::new();
    for _ in 0..syllables {
        w.push_str(ONSET[rng.random_range(0..ONSET.len())]);
        w.push_str(NUCLEUS[rng.random_range(0..NUCLEUS.len())]);
    }
    let mut c = w.chars();
    c.next()
        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
        .unwrap_or_default()
}

/// Card question templates. Each fact is a unique pseudoword pair, so any
/// text overlap between different cards comes only from the shared template
/// - which is exactly what the leakage check measures.
const TEMPLATES: &[&str] = &[
    "What is the capital of {s}?",
    "Which element does the symbol {s} stand for?",
    "What does the enzyme {s} catalyse?",
    "In which year was the Treaty of {s} signed?",
    "What is the currency of {s}?",
];

/// Phase 1: write the synthetic corpus as a real collection file, with
/// deterministic card ids (they seed interval fuzz and the probe coin, and
/// fresh ids come from the wall clock). Probes are attached afterwards by
/// `just probe-gen <path> --deck Default [--baseline]`.
pub fn emit_corpus_collection(cfg: &AblationConfig, path: &Path) -> Result<()> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let mut col = CollectionBuilder::new(path).build()?;
    let notetype = {
        let ntid = col.storage.get_notetype_id("Basic")?.unwrap();
        col.storage.get_notetype(ntid)?.unwrap()
    };
    // The corpus is defined as distinct facts; pseudoword birthday
    // collisions would otherwise give two cards the same answer, and the
    // baseline (Q/A-inversion) generator then emits bit-identical probes for
    // them - which the leakage check rightly flags. (It did: see report §5.)
    let mut used = HashSet::new();
    for i in 0..cfg.cards {
        let mut rng = rng_for(cfg.seed, "corpus", &[i as u64]);
        let mut fresh = |rng: &mut StdRng, syllables| loop {
            let w = pseudoword(rng, syllables);
            if used.insert(w.clone()) {
                return w;
            }
        };
        let subject = fresh(&mut rng, 3);
        let object = fresh(&mut rng, 2);
        let mut note = notetype.new_note();
        *note.fields_mut() = vec![
            TEMPLATES[i % TEMPLATES.len()].replace("{s}", &subject),
            object,
        ];
        col.add_note(&mut note, DeckId(1))?;
    }
    let mut ids: Vec<i64> = all_cards(&col)?.iter().map(|c| c.id.0).collect();
    ids.sort_unstable();
    for (i, id) in ids.iter().enumerate() {
        col.storage.db.execute(
            "update cards set id = ? where id = ?",
            (FIRST_CARD_ID + i as i64, id),
        )?;
    }
    Ok(())
}

/// One card of the corpus as the harness sees it in phase 2, texts included
/// so the leakage checks run against the probes actually stored.
struct CorpusCard {
    card_q: String,
    probes: Vec<(String, String)>, // (question, answer)
}

fn read_corpus(path: &Path) -> Result<Vec<CorpusCard>> {
    let col = CollectionBuilder::new(path).build()?;
    let mut cards = all_cards(&col)?;
    cards.sort_unstable_by_key(|c| c.id);
    cards
        .iter()
        .map(|card| {
            let note = col
                .storage
                .get_note(card.note_id)?
                .or_not_found(card.note_id)?;
            Ok(CorpusCard {
                card_q: note.fields()[0].clone(),
                probes: col
                    .storage
                    .get_probes_for_card(card.id)?
                    .into_iter()
                    .map(|p| (p.question, p.answer))
                    .collect(),
            })
        })
        .collect()
}

/// The simulated learner's ground truth for one card. Surface memory (what
/// the original card cue retrieves) follows the FSRS forgetting-curve shape
/// with its own private stability trajectory; `transfer` scales how much of
/// that retention survives a reworded cue. Rote pairs have low transfer -
/// they remember the card, not the thing.
struct TrueMemory {
    stability: f32,
    last_day: u32,
    growth: f32,
    transfer: f64,
    rote: bool,
}

impl TrueMemory {
    fn retention(&self, day: u32) -> f64 {
        let state = fsrs::MemoryState {
            stability: self.stability,
            difficulty: 5.0,
        };
        fsrs::current_retrievability(
            state,
            (day - self.last_day) as f32,
            fsrs::FSRS5_DEFAULT_DECAY,
        ) as f64
    }

    fn record(&mut self, day: u32, passed: bool) {
        if passed {
            self.stability = (self.stability * self.growth).min(36_500.0);
        } else {
            // failed, then re-studied on the spot
            self.stability = (self.stability * 0.3).max(0.25);
        }
        self.last_day = day;
    }
}

fn new_true_memory(cfg: &AblationConfig, learner: usize, card: usize, day: u32) -> TrueMemory {
    let mut rng = rng_for(cfg.seed, "learner", &[learner as u64, card as u64]);
    let ability: f32 = rng.random_range(0.8..1.25);
    let rote = rng.random_range(0.0..1.0) < cfg.rote_fraction;
    let transfer = if rote {
        rng.random_range(0.05..0.35)
    } else {
        rng.random_range(0.65..0.95)
    };
    TrueMemory {
        stability: rng.random_range(0.5..4.0) * ability,
        last_day: day,
        growth: rng.random_range(1.6..3.2) * ability,
        transfer,
        rote,
    }
}

/// Everything measured about one answered review. `Eq` across the off and
/// stock arms of the same learner is the run-level equivalence check.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Review {
    day: u32,
    card: usize,
    first_exposure: bool,
    /// FSRS short-term (re)learning step - a re-study event minutes after a
    /// failure, not a scheduled recall test. Excluded from recall metrics.
    relearn: bool,
    probe_shown: bool,
    passed: bool,
    /// Scheduler retrievability at serve time, in 1e-6 units (kept integral
    /// so the record is Eq-comparable). None before the first review.
    sched_r_micro: Option<u32>,
    interval_after: u32,
    due_after: i32,
    stability_after_bits: u32,
}

struct ArmRun {
    reviews: Vec<Review>,
    probes_in_collection: usize,
}

fn run_arm(
    cfg: &AblationConfig,
    arm: Arm,
    learner: usize,
    corpus_path: &Path,
    workdir: &Path,
) -> Result<ArmRun> {
    let col_path = workdir.join(format!("ablation-{}-{learner}.anki2", arm.name()));
    if col_path.exists() {
        std::fs::remove_file(&col_path)?;
    }
    std::fs::copy(corpus_path, &col_path)?;
    let mut col = CollectionBuilder::new(&col_path).build()?;
    col.set_config_bool(BoolKey::Fsrs, true, false)?;
    // The load balancer picks intervals from the histogram of what was
    // answered before, making scheduling sensitive to within-day serve
    // order, which the wall clock jitters. Off in every arm, uniformly.
    col.set_config_bool(BoolKey::LoadBalancerEnabled, false, false)?;

    if arm == Arm::Stock {
        // upstream Anki has no probes table content; the serving path
        // short-circuits on the empty table
        col.storage.db.execute("delete from probes", [])?;
    }
    let n_cards = all_cards(&col)?.len();
    let probes_in_collection =
        col.storage
            .db
            .query_row("select count(*) from probes", [], |r| r.get::<_, usize>(0))?;

    // identical config in every arm except probe_rate
    let original = col.get_deck_config(DeckConfigId(1), false)?.unwrap();
    let mut updated = original.clone();
    updated.inner.learn_steps = vec![];
    updated.inner.relearn_steps = vec![];
    updated.inner.new_per_day = 20;
    updated.inner.reviews_per_day = 9999;
    updated.inner.probe_retrievability_threshold = cfg.probe_threshold;
    updated.inner.probe_rate = if arm == Arm::On { cfg.probe_rate } else { 0.0 };
    col.update_deck_config_inner(&mut updated, original, None)?;

    let mut memories: Vec<Option<TrueMemory>> = (0..n_cards).map(|_| None).collect();
    let mut draws: Vec<u64> = vec![0; n_cards];
    let mut reviews = Vec::new();

    for day in 0..cfg.days {
        if day > 0 {
            advance_one_day(&mut col)?;
        }
        col.clear_study_queues();
        let mut answered_today = 0;
        while let Some(queued) = col.get_next_card()? {
            answered_today += 1;
            require!(
                answered_today <= 10_000,
                "runaway day: >10k answers on day {day}"
            );
            let idx = (queued.card.id.0 - FIRST_CARD_ID) as usize;
            let sched_r = scheduler_retrievability(&queued.card);
            let first_exposure = queued.kind == QueueEntryKind::New;
            let relearn = queued.kind == QueueEntryKind::Learning && !first_exposure;

            let (passed, probe_shown) = if first_exposure {
                // First exposure is a study event: the learner reads the
                // answer. Graded Good; true memory starts here.
                memories[idx] = Some(new_true_memory(cfg, learner, idx, day));
                (true, false)
            } else {
                let mem = memories[idx].as_mut().expect("review before exposure");
                let r_true = mem.retention(day);
                let probe_shown = queued.probe.is_some();
                let p = if probe_shown {
                    mem.transfer * r_true
                } else {
                    r_true
                };
                // one seeded draw per (learner, card, event) - arms that
                // schedule identically see identical draws
                let mut rng = rng_for(
                    cfg.seed,
                    "outcome",
                    &[learner as u64, idx as u64, draws[idx]],
                );
                let passed = rng.random_range(0.0..1.0) < p;
                mem.record(day, passed);
                (passed, probe_shown)
            };
            draws[idx] += 1;

            let (rating, new_state) = if passed {
                (Rating::Good, queued.states.good)
            } else {
                (Rating::Again, queued.states.again)
            };
            col.answer_card(&mut CardAnswer {
                card_id: queued.card.id,
                current_state: queued.states.current,
                new_state,
                rating,
                answered_at: TimestampMillis::now(),
                milliseconds_taken: 3000,
                milliseconds_to_reveal: Some(1500),
                variant_id: queued.probe.as_ref().map(|p| p.id),
                custom_data: None,
                from_queue: true,
            })?;

            let after = col.storage.get_card(queued.card.id)?.unwrap();
            reviews.push(Review {
                day,
                card: idx,
                first_exposure,
                relearn,
                probe_shown,
                passed,
                sched_r_micro: sched_r.map(|r| (r * 1e6) as u32),
                interval_after: after.interval,
                // an intraday due is a raw wall-clock timestamp; record only
                // that it was intraday, or the records aren't reproducible
                due_after: if after.due >= 1_000_000 {
                    -1
                } else {
                    after.due
                },
                stability_after_bits: after
                    .memory_state
                    .map(|m| m.stability.to_bits())
                    .unwrap_or(0),
            });
        }
    }

    drop(col);
    let _ = std::fs::remove_file(&col_path);
    // Serve order *within* a day can differ between runs at the margin;
    // per-card chronology is what the equivalence check means, so normalize
    // to (day, card) order (stable: a card's own records keep their order).
    reviews.sort_by_key(|r| (r.day, r.card));
    Ok(ArmRun {
        reviews,
        probes_in_collection,
    })
}

/// Same formula and inputs as `maybe_probe_substitute`, except elapsed time
/// is rounded to whole days: the harness advances time in day units, and
/// rounding keeps the measured value independent of how many wall-clock
/// seconds the run itself takes.
fn scheduler_retrievability(card: &Card) -> Option<f32> {
    let state = card.memory_state?;
    let last = card.last_review_time?;
    let days = (TimestampSecs::now().elapsed_secs_since(last).max(0) as f32 / 86_400.0).round();
    Some(fsrs::current_retrievability(
        state.into(),
        days,
        card.decay.unwrap_or(fsrs::FSRS5_DEFAULT_DECAY),
    ))
}

/// Move the collection one day into the future by shifting its creation
/// stamp and every recorded review time one day into the past. The wall
/// clock never moves, so review timestamps stay consistent with FSRS's
/// elapsed-day arithmetic.
fn advance_one_day(col: &mut Collection) -> Result<()> {
    let crt = col.storage.creation_stamp()?;
    col.set_creation_stamp(TimestampSecs(crt.0 - 86_400))?;
    for mut card in all_cards(col)? {
        let mut dirty = false;
        if let Some(t) = card.last_review_time {
            card.last_review_time = Some(TimestampSecs(t.0 - 86_400));
            dirty = true;
        }
        // intraday (re)learning due dates are raw timestamps and must age
        // with everything else, or a step longer than the learn-ahead limit
        // strands the card forever on our frozen wall clock
        if card.due >= 1_000_000 {
            card.due -= 86_400;
            dirty = true;
        }
        if dirty {
            col.storage.update_card(&card)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// metrics

fn wilson_ci(passes: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let z = 1.96f64;
    let (p, n) = (passes as f64 / n as f64, n as f64);
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let half = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / denom;
    (centre - half, centre + half)
}

struct Calibration {
    n: usize,
    brier: f64,
    log_loss: f64,
    base_rate: f64,
    base_rate_brier: f64,
    mean_forecast: f64,
}

fn calibration(rows: &[(f64, bool)]) -> Option<Calibration> {
    if rows.is_empty() {
        return None;
    }
    // summation order must not depend on the order reviews were served in
    let mut rows = rows.to_vec();
    rows.sort_unstable_by_key(|(f, o)| (f.to_bits(), *o));
    let n = rows.len() as f64;
    let base_rate = rows.iter().filter(|r| r.1).count() as f64 / n;
    let mut brier = 0.0;
    let mut log_loss = 0.0;
    let mut base_brier = 0.0;
    let mut mean_forecast = 0.0;
    for &(f, outcome) in &rows {
        let y = if outcome { 1.0 } else { 0.0 };
        brier += (f - y) * (f - y);
        let fc = f.clamp(1e-6, 1.0 - 1e-6);
        log_loss -= y * fc.ln() + (1.0 - y) * (1.0 - fc).ln();
        base_brier += (base_rate - y) * (base_rate - y);
        mean_forecast += f;
    }
    Some(Calibration {
        n: rows.len(),
        brier: brier / n,
        log_loss: log_loss / n,
        base_rate,
        base_rate_brier: base_brier / n,
        mean_forecast: mean_forecast / n,
    })
}

fn calibration_json(c: &Option<Calibration>) -> Value {
    match c {
        None => Value::Null,
        Some(c) => json!({
            "n": c.n,
            "brier": c.brier,
            "log_loss": c.log_loss,
            "outcome_base_rate": c.base_rate,
            "base_rate_brier": c.base_rate_brier,
            "mean_forecast": c.mean_forecast,
        }),
    }
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn shingles(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split(' ').collect();
    if words.len() < 3 {
        return vec![text.to_string()];
    }
    words.windows(3).map(|w| w.join(" ")).collect()
}

fn jaccard(a: &str, b: &str) -> f64 {
    let sa: HashSet<String> = shingles(a).into_iter().collect();
    let sb: HashSet<String> = shingles(b).into_iter().collect();
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// The leakage check from data/mcat-taxonomy/report.md §5.3, adapted to this
/// corpus and run against the probe texts actually stored by the generation
/// pipeline. Held-back items are the card and probe texts of held-back
/// cards.
///
/// - Risk A (contamination): a probe of a non-held-back card duplicating a
///   held-back item. Gate: zero exact duplicates and zero near-duplicates of a
///   held-back *card* question. Near-duplicates against held-back *probe*
///   questions are reported separately rather than gated: generators that fill
///   a fixed template (the no-AI baseline does) make every probe
///   surface-similar to every other probe, which is a property of the
///   generator, not evidence that a held-back fact leaked - the facts are
///   distinct pseudoword pairs and fact identity is carried by the exact text.
/// - Risk B (synonym swap): minimum distance from each probe to its own source
///   card; the pipeline's `gate_reject_reason` enforces this at generation
///   time, this re-measures it on what was stored.
/// - Risk C (answer in the stem): count of stored probes whose normalized
///   answer appears inside their own question; the pipeline gates this too.
fn leakage_check(corpus: &[CorpusCard], heldback: &[bool]) -> Value {
    let held_cards: Vec<String> = corpus
        .iter()
        .zip(heldback)
        .filter(|(_, &h)| h)
        .map(|(c, _)| normalize(&c.card_q))
        .collect();
    let held_probes: Vec<String> = corpus
        .iter()
        .zip(heldback)
        .filter(|(_, &h)| h)
        .flat_map(|(c, _)| c.probes.iter().map(|(q, _)| normalize(q)))
        .collect();
    let held_exact: HashSet<&String> = held_cards.iter().chain(held_probes.iter()).collect();

    let mut pairs = 0usize;
    let mut exact = 0usize;
    let mut near_dup_vs_held_cards = 0usize;
    let mut near_dup_vs_held_probes = 0usize;
    let mut max_j_cards: f64 = 0.0;
    let mut max_j_probes: f64 = 0.0;
    let mut min_own_dist: f64 = 1.0;
    let mut own_answer_leaks = 0usize;
    for (i, card) in corpus.iter().enumerate() {
        for (q, a) in &card.probes {
            let probe = normalize(q);
            // risk B/C: against its own card, regardless of split
            min_own_dist = min_own_dist.min(1.0 - jaccard(&probe, &normalize(&card.card_q)));
            if probe.contains(&normalize(a)) {
                own_answer_leaks += 1;
            }
            if heldback[i] {
                continue;
            }
            // risk A: against the held-back set
            if held_exact.contains(&probe) {
                exact += 1;
            }
            for item in &held_cards {
                pairs += 1;
                let j = jaccard(&probe, item);
                max_j_cards = max_j_cards.max(j);
                if j >= 0.8 {
                    near_dup_vs_held_cards += 1;
                }
            }
            for item in &held_probes {
                pairs += 1;
                let j = jaccard(&probe, item);
                max_j_probes = max_j_probes.max(j);
                if j >= 0.8 {
                    near_dup_vs_held_probes += 1;
                }
            }
        }
    }
    json!({
        "method": "normalized exact match + Jaccard over 3-word shingles; every non-held-back probe vs every held-back item (card and probe text), plus per-probe checks against its own source card",
        "pairs_checked": pairs,
        "exact_duplicates": exact,
        "near_dup_vs_heldback_cards_ge_0.8": near_dup_vs_held_cards,
        "max_jaccard_vs_heldback_cards": max_j_cards,
        "near_dup_vs_heldback_probes_ge_0.8": near_dup_vs_held_probes,
        "max_jaccard_vs_heldback_probes": max_j_probes,
        "min_probe_to_own_card_distance": min_own_dist,
        "probes_containing_own_answer": own_answer_leaks,
        "passed": exact == 0 && near_dup_vs_held_cards == 0 && own_answer_leaks == 0,
    })
}

// ---------------------------------------------------------------------------
// entry point

/// Phase 2: run all three arms against the corpus collection (which must
/// already contain probes) and return the measured results. Temporary
/// per-arm collection copies are made next to the corpus file.
pub fn run(cfg: &AblationConfig, corpus_path: &Path) -> Result<Value> {
    let workdir = corpus_path.parent().unwrap_or(Path::new(".")).to_owned();
    let corpus = read_corpus(corpus_path)?;
    let n_cards = corpus.len();
    require!(n_cards > 0, "corpus collection has no cards");
    let probe_generators: Vec<String> = {
        let col = CollectionBuilder::new(corpus_path).build()?;
        let mut gens: Vec<String> = col
            .storage
            .db
            .prepare("select provenance from probes")?
            .query_map([], |r| r.get::<_, String>(0))?
            .filter_map(|p| {
                serde_json::from_str::<Value>(&p.ok()?)
                    .ok()?
                    .get("generator")?
                    .as_str()
                    .map(str::to_owned)
            })
            .collect();
        gens.sort();
        gens.dedup();
        gens
    };

    let heldback: Vec<bool> = (0..n_cards).map(|i| i % cfg.heldback_every == 0).collect();
    let rote: Vec<Vec<bool>> = (0..cfg.learners)
        .map(|l| {
            (0..n_cards)
                .map(|c| new_true_memory(cfg, l, c, 0).rote)
                .collect()
        })
        .collect();

    let mut runs: Vec<(Arm, usize, ArmRun)> = Vec::new();
    for &arm in &[Arm::Stock, Arm::Off, Arm::On] {
        for learner in 0..cfg.learners {
            runs.push((
                arm,
                learner,
                run_arm(cfg, arm, learner, corpus_path, &workdir)?,
            ));
        }
    }

    // --- arm equivalence: off must equal stock, review for review ---------
    let mut equivalent = true;
    let mut first_divergence = Value::Null;
    for learner in 0..cfg.learners {
        let stock = &runs
            .iter()
            .find(|r| r.0 == Arm::Stock && r.1 == learner)
            .unwrap()
            .2;
        let off = &runs
            .iter()
            .find(|r| r.0 == Arm::Off && r.1 == learner)
            .unwrap()
            .2;
        if stock.reviews != off.reviews {
            equivalent = false;
            if first_divergence.is_null() {
                let i = stock
                    .reviews
                    .iter()
                    .zip(&off.reviews)
                    .position(|(a, b)| a != b)
                    .unwrap_or(stock.reviews.len().min(off.reviews.len()));
                first_divergence = json!({
                    "learner": learner,
                    "review_index": i,
                    "stock": format!("{:?}", stock.reviews.get(i)),
                    "off": format!("{:?}", off.reviews.get(i)),
                });
            }
        }
    }

    // --- pooled measurement rows ------------------------------------------
    let mut on_high_r_orig = (0usize, 0usize); // (passes, n) original shown, R >= threshold
    let mut on_high_r_probe = (0usize, 0usize);
    let mut off_card_rows_held: Vec<(f64, bool)> = Vec::new();
    let mut on_probe_rows_held: Vec<(f64, bool)> = Vec::new();
    let mut on_orig_rows_held: Vec<(f64, bool)> = Vec::new();
    let mut off_by_rote = [(0usize, 0usize); 2]; // [understood, rote] = (passes, n)
    let mut on_probe_by_rote = [(0usize, 0usize); 2];
    let mut totals: Vec<Value> = Vec::new();

    for &arm in &[Arm::Stock, Arm::Off, Arm::On] {
        let mut n_reviews = 0;
        let mut n_probe = 0;
        let mut n_pass = 0;
        let mut n_relearn = 0;
        let mut probes_in_collection = 0;
        for (a, learner, run) in &runs {
            if *a != arm {
                continue;
            }
            probes_in_collection = run.probes_in_collection;
            for r in &run.reviews {
                if r.first_exposure {
                    continue;
                }
                if r.relearn {
                    // re-study step minutes after a failure, not a recall test
                    n_relearn += 1;
                    continue;
                }
                n_reviews += 1;
                n_pass += r.passed as usize;
                n_probe += r.probe_shown as usize;
                let held = heldback[r.card];
                let is_rote = rote[*learner][r.card] as usize;
                let sched_r = r.sched_r_micro.map(|m| m as f64 / 1e6);
                match arm {
                    Arm::Off => {
                        off_by_rote[is_rote].0 += r.passed as usize;
                        off_by_rote[is_rote].1 += 1;
                        if held {
                            if let Some(f) = sched_r {
                                off_card_rows_held.push((f, r.passed));
                            }
                        }
                    }
                    Arm::On => {
                        let high_r = sched_r.is_some_and(|f| f >= cfg.probe_threshold as f64);
                        if r.probe_shown {
                            on_probe_by_rote[is_rote].0 += r.passed as usize;
                            on_probe_by_rote[is_rote].1 += 1;
                        }
                        if high_r {
                            let slot = if r.probe_shown {
                                &mut on_high_r_probe
                            } else {
                                &mut on_high_r_orig
                            };
                            slot.0 += r.passed as usize;
                            slot.1 += 1;
                        }
                        if held {
                            if let Some(f) = sched_r {
                                if r.probe_shown {
                                    on_probe_rows_held.push((f, r.passed));
                                } else {
                                    on_orig_rows_held.push((f, r.passed));
                                }
                            }
                        }
                    }
                    Arm::Stock => {}
                }
            }
        }
        totals.push(json!({
            "arm": arm.name(),
            "learners": cfg.learners,
            "probes_in_collection": probes_in_collection,
            "recall_reviews": n_reviews,
            "relearning_steps": n_relearn,
            "probe_served_reviews": n_probe,
            "pass_rate": if n_reviews > 0 { n_pass as f64 / n_reviews as f64 } else { f64::NAN },
        }));
    }

    // --- the gap ------------------------------------------------------------
    let p_orig = on_high_r_orig.0 as f64 / on_high_r_orig.1.max(1) as f64;
    let p_probe = on_high_r_probe.0 as f64 / on_high_r_probe.1.max(1) as f64;
    let gap = p_orig - p_probe;
    let se = (p_orig * (1.0 - p_orig) / on_high_r_orig.1.max(1) as f64
        + p_probe * (1.0 - p_probe) / on_high_r_probe.1.max(1) as f64)
        .sqrt();
    let gap_ci = (gap - 1.96 * se, gap + 1.96 * se);

    let cal_card = calibration(&off_card_rows_held);
    let cal_probe = calibration(&on_probe_rows_held);
    let cal_on_orig = calibration(&on_orig_rows_held);

    let leakage = leakage_check(&corpus, &heldback);
    // fair-testing gate: contaminated held-back items invalidate the run,
    // exactly like an off/stock scheduling divergence does
    let run_valid = equivalent && leakage["passed"] == json!(true);

    let falsified_gap = !(gap >= 0.05 && gap_ci.0 > 0.0);
    let falsified_brier = match (&cal_probe, &cal_card) {
        (Some(p), Some(c)) => p.brier <= c.brier,
        _ => true, // no data is not evidence for the claim
    };

    let rate = |t: (usize, usize)| t.0 as f64 / t.1.max(1) as f64;
    Ok(json!({
        "disclaimer": "All learners are SIMULATED. Probe text cannot influence simulated outcomes; it is exercised for serving and text-level checks only. No human data.",
        "probe_generators_in_corpus": probe_generators,
        "config": {
            "seed": cfg.seed,
            "learners_per_arm": cfg.learners,
            "cards": n_cards,
            "days": cfg.days,
            "probe_rate_on_arm": cfg.probe_rate,
            "probe_retrievability_threshold": cfg.probe_threshold,
            "rote_fraction": cfg.rote_fraction,
            "heldback_every_nth_card": cfg.heldback_every,
            "fsrs_parameters": "stock defaults, never fitted",
        },
        "falsification_condition": FALSIFICATION,
        "arms": totals,
        "arm_equivalence_off_vs_stock": {
            "identical_review_sequences": equivalent,
            "first_divergence": first_divergence,
        },
        "gap": {
            "description": "on-arm reviews with scheduler retrievability >= threshold: pass rate when the original card was shown vs when the probe was shown",
            "original": {"passes": on_high_r_orig.0, "n": on_high_r_orig.1, "pass_rate": p_orig,
                          "wilson95": wilson_ci(on_high_r_orig.0, on_high_r_orig.1)},
            "probe": {"passes": on_high_r_probe.0, "n": on_high_r_probe.1, "pass_rate": p_probe,
                       "wilson95": wilson_ci(on_high_r_probe.0, on_high_r_probe.1)},
            "gap": gap,
            "gap_ci95": [gap_ci.0, gap_ci.1],
        },
        "rote_blindness": {
            "description": "pass rates for rote vs understood (learner,card) pairs; card-only reviews cannot separate them, probe outcomes can",
            "off_arm_card_reviews": {
                "understood_pass_rate": rate(off_by_rote[0]), "understood_n": off_by_rote[0].1,
                "rote_pass_rate": rate(off_by_rote[1]), "rote_n": off_by_rote[1].1,
            },
            "on_arm_probe_reviews": {
                "understood_pass_rate": rate(on_probe_by_rote[0]), "understood_n": on_probe_by_rote[0].1,
                "rote_pass_rate": rate(on_probe_by_rote[1]), "rote_n": on_probe_by_rote[1].1,
            },
        },
        "calibration_heldback": {
            "description": "FSRS retrievability as forecast, scored only on held-back cards (never used for any tuning)",
            "card_outcomes_off_arm": calibration_json(&cal_card),
            "original_outcomes_on_arm": calibration_json(&cal_on_orig),
            "probe_outcomes_on_arm": calibration_json(&cal_probe),
        },
        "leakage_check": leakage,
        "verdict": {
            "run_valid": run_valid,
            "falsified_by_gap": falsified_gap,
            "falsified_by_brier": falsified_brier,
            "claim_supported_in_simulation": run_valid && !falsified_gap && !falsified_brier,
        },
    }))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::probe::Probe;

    #[test]
    fn leakage_checker_flags_a_planted_duplicate() {
        let corpus = vec![
            CorpusCard {
                card_q: "What is the capital of Xanadu?".into(),
                // duplicates the held-back card's probe text exactly
                probes: vec![(
                    "Which city serves as the seat of government of Kubla?".into(),
                    "A".into(),
                )],
            },
            CorpusCard {
                card_q: "What is the capital of Kubla?".into(),
                probes: vec![(
                    "Which city serves as the seat of government of Kubla?".into(),
                    "B".into(),
                )],
            },
        ];
        let v = leakage_check(&corpus, &[false, true]);
        assert_eq!(v["exact_duplicates"], 1);
        assert_eq!(v["passed"], false);

        // distinct subjects: templates overlap but nothing duplicates exactly
        let corpus2 = vec![
            CorpusCard {
                card_q: "What is the capital of Xanadu?".into(),
                probes: vec![(
                    "Which city serves as the seat of government of Xanadu?".into(),
                    "A".into(),
                )],
            },
            corpus.into_iter().nth(1).unwrap(),
        ];
        let v2 = leakage_check(&corpus2, &[false, true]);
        assert_eq!(v2["exact_duplicates"], 0);
        // template similarity against a held-back probe is measured, not lost
        assert!(v2["max_jaccard_vs_heldback_probes"].as_f64().unwrap() > 0.7);
    }

    /// The reproducibility claim: same seed, same numbers, and the off arm
    /// schedules identically to stock at run scale. Probes here are attached
    /// directly rather than through the Python pipeline (unit tests cannot
    /// shell out to it); `run` treats stored probes identically either way.
    #[test]
    fn tiny_run_is_deterministic_and_off_equals_stock() {
        let cfg = AblationConfig {
            learners: 2,
            cards: 20,
            days: 25,
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let corpus_path = dir.path().join("corpus.anki2");
        emit_corpus_collection(&cfg, &corpus_path).unwrap();
        {
            let mut col = CollectionBuilder::new(&corpus_path).build().unwrap();
            for i in 0..cfg.cards {
                let mut probe = Probe {
                    card_id: CardId(FIRST_CARD_ID + i as i64),
                    question: format!("test probe question {i}"),
                    answer: format!("test answer {i}"),
                    citation: format!("card {i}"),
                    provenance: r#"{"generator":"unit-test"}"#.into(),
                    ..Default::default()
                };
                col.add_probe(&mut probe).unwrap();
            }
        }

        let a = run(&cfg, &corpus_path).unwrap();
        let b = run(&cfg, &corpus_path).unwrap();
        assert_eq!(
            a["arm_equivalence_off_vs_stock"]["identical_review_sequences"], true,
            "{}",
            a["arm_equivalence_off_vs_stock"]
        );
        // wall-clock-derived values must not leak into the results
        assert_eq!(a, b);
        // probes were actually served in the on arm
        let on = &a["arms"][2];
        assert_eq!(on["arm"], "on");
        assert!(on["probe_served_reviews"].as_u64().unwrap() > 0);
        assert_eq!(a["probe_generators_in_corpus"], json!(["unit-test"]));
    }
}
