//! Motif-level decoding: a molecule as a sequence of whole ring systems,
//! groups and single atoms.
//!
//! The completion model builds a graph one atom at a time. Here the alphabet
//! is coarser: `tools/ms2/motif_tokens.py` cuts every training molecule into
//! motifs (ring systems, acyclic groups, single atoms) that form a tree, and
//! writes it as
//!
//! ```text
//! molecule := MOTIF m  body  END
//! body     := ( ATOM a  BOND o  MOTIF m'  ATOM b  body  END )*
//! ```
//!
//! `ATOM a BOND o MOTIF m' ATOM b` attaches motif `m'` by a bond of order `o`
//! from atom `a` of the motif on top of the stack to atom `b` of the new one
//! and pushes it; `END` pops. This module holds the host side: the
//! vocabulary ([`MotifVocab`]), the stack machine that reads the tokens and
//! decides which one may come next ([`MotifMachine`]), the token ids of the
//! sequence model ([`Layout`]) and the conditioning prefix
//! ([`Layout::prefix`]) and the search ([`beam_search`]). The sequence model
//! itself is a plain [`Mamba3Lm`](crate::models::lm::Mamba3Lm);
//! `examples/ms2_motif_decoder.rs` trains it and searches it with
//! [`MotifMachine::allowed`] as the mask.
//!
//! The machine is specified token by token so that a mask can be read off it
//! and so that it can be stated exactly elsewhere: `lean/MotifDecoder` holds
//! the same machine in Lean 4 with proofs of what it guarantees for a
//! vocabulary whose motifs are well-formed and connected, which is what
//! [`MotifVocab`] checks on load: every accepted sequence is a connected
//! graph, no atom spends more bond order than it has, the number of bonds is
//! the motifs' own plus one per attachment (a counting identity; cycles are
//! not formalised), and with a budget the formula is exact. The proofs are
//! about the Lean definitions: this implementation is tested against the Lean
//! executable's output (`tests/ms2_motif.rs`, `examples/ms2_motif_check.rs`),
//! not proven equal to it, and it adds what the model does not have — a
//! bound on the graph ([`MAX_GRAPH_ATOMS`]), the pruning rule
//! [`MotifMachine::dead`] and the search.

use std::path::Path;

use cubecl::prelude::Runtime;
use serde::Deserialize;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::models::hybrid::LayerCache;
use crate::models::mamba3::MixerCache;
use crate::nn::attention::AttentionCache;
use crate::ssm::SsmState;
use crate::tensor::ops::index::{IdTensor, check_gather_ids, gather_rows};
use crate::tensor::{Shape, Tensor};

use super::completion_fingerprint::SparseFingerprint;

/// Padding id; also the loss's ignore index.
pub const PAD: u32 = 0;
/// `END`: pop the open motif.
pub const END: u32 = 1;
/// `BOND o` is `BOND_BASE + o` for `o` in `1..=3`.
pub const BOND_BASE: u32 = 1;
/// `ATOM a` is `ATOM_BASE + a`.
pub const ATOM_BASE: u32 = 5;
/// `MOTIF m` is `MOTIF_BASE + m`.
pub const MOTIF_BASE: u32 = 37;
/// Atoms a motif may hold (the number of `ATOM` ids).
pub const MAX_MOTIF_ATOMS: usize = 32;
/// Atoms a graph may hold. The machine refuses a motif that would pass it,
/// which keeps every index in `u16`, every count in `u16` and `free_sum`
/// below `u16::MAX` (256 atoms of at most 255 each). The Lean model has no
/// such bound; the two agree on every sequence that stays under it.
pub const MAX_GRAPH_ATOMS: usize = 256;
/// Motifs a vocabulary may hold (keeps every token id far inside `u32`).
pub const MAX_MOTIFS: usize = 1 << 20;
/// Elements of a formula, in the order of every composition array:
/// C H N O F P S Cl Br I (atomic numbers).
pub const ELEMENTS: [u8; 10] = [6, 1, 7, 8, 9, 15, 16, 17, 35, 53];
/// Index of hydrogen in [`ELEMENTS`].
pub const HYDROGEN: usize = 1;
/// Largest element count a formula token can carry.
pub const MAX_COUNT: u16 = 127;
/// Fingerprint tokens a prefix holds at most.
pub const FINGERPRINT_TOKENS: usize = 128;
/// Longest conditioning prefix: BOS, one count token per element, the
/// fingerprint tokens with two level separators, SEP.
pub const PREFIX_LEN: usize = 1 + 10 + FINGERPRINT_TOKENS + 2 + 1;

/// Element counts in [`ELEMENTS`] order.
pub type Formula = [u16; 10];

/// One motif of the vocabulary.
#[derive(Clone, Debug)]
pub struct Motif {
    /// Element index (into [`ELEMENTS`]) of each atom.
    pub elements: Vec<u8>,
    /// Bond order each atom can still spend on attachments.
    pub free: Vec<u8>,
    /// Bonds inside the motif, `(i, j, order)`.
    pub bonds: Vec<(u8, u8, u8)>,
    /// Heavy-atom counts (the hydrogen entry is 0).
    pub composition: Formula,
    /// Sum of `free`.
    pub free_sum: u32,
    /// Largest entry of `free`.
    pub max_free: u8,
}

#[derive(Deserialize)]
struct RawMotif {
    elements: Vec<u8>,
    free: Vec<u8>,
    bonds: Vec<(u8, u8, u8)>,
}

#[derive(Deserialize)]
struct RawVocab {
    format: String,
    elements: Vec<u8>,
    motifs: Vec<RawMotif>,
}

/// The motif vocabulary written by `tools/ms2/motif_tokens.py prepare`.
#[derive(Clone, Debug)]
pub struct MotifVocab {
    /// Motifs in id order.
    pub motifs: Vec<Motif>,
}

impl MotifVocab {
    /// Load `lm_vocab.json`.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_json(&std::fs::read_to_string(path)?)
    }

    /// Load from JSON text. Requires format `motif_lm_vocab_v1`, the element
    /// order of [`ELEMENTS`], at most [`MAX_MOTIFS`] motifs, and for every
    /// motif: 1 to [`MAX_MOTIF_ATOMS`] atoms, as many `free` entries as
    /// atoms, known elements other than hydrogen, bonds with distinct
    /// in-range ends and an order in `1..=3`, and all atoms connected through
    /// those bonds. Anything else is [`Error::Config`].
    pub fn load_json(text: &str) -> Result<Self> {
        let raw: RawVocab = serde_json::from_str(text)?;
        if raw.format != "motif_lm_vocab_v1" {
            return Err(Error::config(format!(
                "MotifVocab::load: format {:?} is not \"motif_lm_vocab_v1\"",
                raw.format
            )));
        }
        if raw.elements != ELEMENTS {
            return Err(Error::config(format!(
                "MotifVocab::load: element order {:?} is not {ELEMENTS:?}",
                raw.elements
            )));
        }
        Self::from_parts(
            raw.motifs
                .into_iter()
                .map(|motif| (motif.elements, motif.free, motif.bonds))
                .collect(),
        )
    }

    /// Build a vocabulary from `(atomic numbers, free, bonds)` per motif,
    /// with the checks [`load_json`](Self::load_json) lists.
    pub fn from_parts(parts: Vec<(Vec<u8>, Vec<u8>, Vec<(u8, u8, u8)>)>) -> Result<Self> {
        let bad = |what: String| Error::config(format!("MotifVocab: {what}"));
        let mut motifs = Vec::with_capacity(parts.len());
        for (m, (atomic, free, bonds)) in parts.into_iter().enumerate() {
            let n = atomic.len();
            if n == 0 || n > MAX_MOTIF_ATOMS || free.len() != n {
                return Err(bad(format!(
                    "motif {m} has {n} atoms and {} free entries",
                    free.len()
                )));
            }
            let mut composition = [0u16; 10];
            let mut elements = Vec::with_capacity(n);
            for &z in &atomic {
                let e = ELEMENTS
                    .iter()
                    .position(|&known| known == z)
                    .filter(|&e| e != HYDROGEN)
                    .ok_or_else(|| bad(format!("motif {m} holds element {z}")))?;
                composition[e] += 1;
                elements.push(e as u8);
            }
            // Union-find over the bonds: a motif must be one connected piece,
            // which is the hypothesis the connectivity proof needs.
            let mut parent: Vec<usize> = (0..n).collect();
            fn find(parent: &mut [usize], mut x: usize) -> usize {
                while parent[x] != x {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                x
            }
            for &(i, j, order) in &bonds {
                if i == j || usize::from(i) >= n || usize::from(j) >= n || !(1..=3).contains(&order)
                {
                    return Err(bad(format!("motif {m} has the bond ({i}, {j}, {order})")));
                }
                let (a, b) = (find(&mut parent, usize::from(i)), find(&mut parent, usize::from(j)));
                parent[a] = b;
            }
            let root = find(&mut parent, 0);
            if (1..n).any(|i| find(&mut parent, i) != root) {
                return Err(bad(format!("motif {m} is not connected")));
            }
            motifs.push(Motif {
                free_sum: free.iter().map(|&f| u32::from(f)).sum(),
                max_free: free.iter().copied().max().unwrap_or(0),
                elements,
                free,
                bonds,
                composition,
            });
        }
        if motifs.is_empty() || motifs.len() > MAX_MOTIFS {
            return Err(bad(format!("{} motifs (needs 1 to {MAX_MOTIFS})", motifs.len())));
        }
        Ok(Self { motifs })
    }

    /// Number of motifs.
    pub fn len(&self) -> usize {
        self.motifs.len()
    }

    /// Whether the vocabulary is empty (never, for a loaded one).
    pub fn is_empty(&self) -> bool {
        self.motifs.is_empty()
    }
}

/// Where the machine is inside one attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Nothing read yet: the root motif comes next.
    Start,
    /// Between attachments: `ATOM a` opens one, `END` pops.
    Body,
    /// `ATOM a` read.
    AfterAtom(u8),
    /// `ATOM a BOND o` read.
    AfterBond(u8, u8),
    /// `ATOM a BOND o MOTIF m` read: the entry atom comes next.
    AfterMotif(u8, u8, u32),
    /// The root was popped.
    Done,
}

/// The stack machine that reads a motif sequence and builds the graph.
///
/// [`allowed`](Self::allowed) is the decoder's mask and
/// [`apply`](Self::apply) the transition; `apply` on a token that is not
/// allowed is an error and leaves the machine unchanged.
#[derive(Clone, Debug)]
pub struct MotifMachine {
    /// Element index of every atom added so far.
    pub elements: Vec<u8>,
    /// Bond order every atom can still spend.
    pub free: Vec<u8>,
    /// Bonds in the order they were added, `(i, j, order)`.
    pub bonds: Vec<(u16, u16, u8)>,
    /// Open motifs as `(first atom, atoms)`, the top last.
    pub stack: Vec<(u16, u16)>,
    /// Position inside the grammar.
    pub phase: Phase,
    /// Heavy-atom counts so far (the hydrogen entry stays 0).
    pub counts: Formula,
    /// Sum of `free`: the hydrogens the graph would carry if closed now.
    pub free_sum: u32,
    /// Motifs added.
    pub motifs: u32,
}

impl Default for MotifMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl MotifMachine {
    /// The initial state.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            free: Vec::new(),
            bonds: Vec::new(),
            stack: Vec::new(),
            phase: Phase::Start,
            counts: [0; 10],
            free_sum: 0,
            motifs: 0,
        }
    }

    /// Whether the sequence is complete.
    pub fn done(&self) -> bool {
        self.phase == Phase::Done
    }

    /// The formula of the graph as it stands: heavy atoms plus the hydrogens
    /// that `free` leaves (exact: [`MAX_GRAPH_ATOMS`] keeps the sum in range).
    pub fn formula(&self) -> Formula {
        let mut out = self.counts;
        out[HYDROGEN] = u16::try_from(self.free_sum).expect("free_sum is bounded by MAX_GRAPH_ATOMS");
        out
    }

    fn fits(&self, motif: &Motif, budget: Option<&Formula>) -> bool {
        if self.elements.len() + motif.elements.len() > MAX_GRAPH_ATOMS {
            return false;
        }
        match budget {
            None => true,
            // Counts stay at or below MAX_GRAPH_ATOMS, so the sum cannot wrap.
            Some(target) => (0..10)
                .all(|e| e == HYDROGEN || self.counts[e] + motif.composition[e] <= target[e]),
        }
    }

    fn top_free(&self, a: u8) -> Option<u8> {
        let &(start, size) = self.stack.last()?;
        (u16::from(a) < size).then(|| self.free[usize::from(start) + usize::from(a)])
    }

    /// Whether `token` may come next. `budget` adds the exact-formula rule:
    /// a motif must fit the heavy atoms still missing, and the `END` that
    /// empties the stack needs every heavy count met and `free_sum` equal to
    /// the target's hydrogens. With or without a budget, a motif that would
    /// take the graph past [`MAX_GRAPH_ATOMS`] is refused.
    pub fn allowed(&self, vocab: &MotifVocab, budget: Option<&Formula>, token: u32) -> bool {
        let motif_of = |token: u32| {
            token
                .checked_sub(MOTIF_BASE)
                .and_then(|m| vocab.motifs.get(m as usize))
        };
        let atom_of = |token: u32| {
            (ATOM_BASE..ATOM_BASE + MAX_MOTIF_ATOMS as u32)
                .contains(&token)
                .then(|| (token - ATOM_BASE) as u8)
        };
        let bond_of =
            |token: u32| (BOND_BASE + 1..=BOND_BASE + 3).contains(&token).then(|| (token - BOND_BASE) as u8);
        match self.phase {
            Phase::Start => motif_of(token).is_some_and(|motif| self.fits(motif, budget)),
            Phase::Body => {
                if token == END {
                    return match budget {
                        Some(target) if self.stack.len() == 1 => {
                            (0..10).all(|e| e == HYDROGEN || self.counts[e] == target[e])
                                && self.free_sum == u32::from(target[HYDROGEN])
                        }
                        _ => true,
                    };
                }
                atom_of(token).is_some_and(|a| self.top_free(a).is_some_and(|f| f >= 1))
            }
            Phase::AfterAtom(a) => {
                bond_of(token).is_some_and(|o| self.top_free(a).is_some_and(|f| f >= o))
            }
            Phase::AfterBond(_, o) => {
                motif_of(token).is_some_and(|motif| motif.max_free >= o && self.fits(motif, budget))
            }
            Phase::AfterMotif(_, o, m) => atom_of(token).is_some_and(|b| {
                vocab.motifs[m as usize]
                    .free
                    .get(usize::from(b))
                    .is_some_and(|&f| f >= o)
            }),
            Phase::Done => false,
        }
    }

    /// Every token [`allowed`](Self::allowed) accepts, in id order.
    pub fn allowed_tokens(&self, vocab: &MotifVocab, budget: Option<&Formula>, out: &mut Vec<u32>) {
        out.clear();
        let range = match self.phase {
            Phase::Start | Phase::AfterBond(..) => MOTIF_BASE..MOTIF_BASE + vocab.len() as u32,
            Phase::Body => END..ATOM_BASE + MAX_MOTIF_ATOMS as u32,
            Phase::AfterAtom(_) => BOND_BASE + 1..BOND_BASE + 4,
            Phase::AfterMotif(..) => ATOM_BASE..ATOM_BASE + MAX_MOTIF_ATOMS as u32,
            Phase::Done => 0..0,
        };
        out.extend(range.filter(|&token| self.allowed(vocab, budget, token)));
    }

    /// Whether no continuation can be accepted under `budget`. Once every
    /// heavy atom is placed no motif fits any more, so `free_sum` is final:
    /// the state is dead if that is not the target's hydrogen count, or if
    /// an attachment is already open. A search may drop such a state; it
    /// never drops one that could still be completed.
    pub fn dead(&self, budget: &Formula) -> bool {
        let placed = (0..10).all(|e| e == HYDROGEN || self.counts[e] == budget[e]);
        match self.phase {
            // Closing is all that is left, and the last `END` will be refused.
            Phase::Body => placed && self.free_sum != u32::from(budget[HYDROGEN]),
            // An attachment is open and no motif can complete it.
            Phase::AfterAtom(_) | Phase::AfterBond(..) => placed,
            Phase::Start | Phase::AfterMotif(..) | Phase::Done => false,
        }
    }

    fn push_motif(&mut self, motif: &Motif) -> u16 {
        let start = self.elements.len() as u16;
        self.elements.extend_from_slice(&motif.elements);
        self.free.extend_from_slice(&motif.free);
        for &(i, j, order) in &motif.bonds {
            self.bonds
                .push((start + u16::from(i), start + u16::from(j), order));
        }
        for e in 0..10 {
            self.counts[e] += motif.composition[e];
        }
        self.free_sum += motif.free_sum;
        self.motifs += 1;
        self.stack.push((start, motif.elements.len() as u16));
        start
    }

    /// Read `token`. [`Error::Config`] when it is not allowed (the machine
    /// is then unchanged); `budget` is the one given to
    /// [`allowed`](Self::allowed).
    pub fn apply(&mut self, vocab: &MotifVocab, budget: Option<&Formula>, token: u32) -> Result<()> {
        if !self.allowed(vocab, budget, token) {
            return Err(Error::config(format!(
                "MotifMachine::apply: token {token} is not allowed in {:?}",
                self.phase
            )));
        }
        match self.phase {
            Phase::Start => {
                self.push_motif(&vocab.motifs[(token - MOTIF_BASE) as usize]);
                self.phase = Phase::Body;
            }
            Phase::Body if token == END => {
                self.stack.pop();
                self.phase = if self.stack.is_empty() {
                    Phase::Done
                } else {
                    Phase::Body
                };
            }
            Phase::Body => self.phase = Phase::AfterAtom((token - ATOM_BASE) as u8),
            Phase::AfterAtom(a) => self.phase = Phase::AfterBond(a, (token - BOND_BASE) as u8),
            Phase::AfterBond(a, o) => self.phase = Phase::AfterMotif(a, o, token - MOTIF_BASE),
            Phase::AfterMotif(a, o, m) => {
                let b = (token - ATOM_BASE) as u16;
                let (parent, _) = *self.stack.last().expect("body phases keep a frame");
                let source = parent + u16::from(a);
                let start = self.push_motif(&vocab.motifs[m as usize]);
                let target = start + b;
                self.bonds.push((source, target, o));
                self.free[usize::from(source)] -= o;
                self.free[usize::from(target)] -= o;
                self.free_sum -= 2 * u32::from(o);
                self.phase = Phase::Body;
            }
            Phase::Done => unreachable!("nothing is allowed after the end"),
        }
        Ok(())
    }

    /// Run a whole sequence from the initial state. `Ok(machine)` when every
    /// token was allowed and the sequence is complete, otherwise the index of
    /// the first rejected token (the sequence length if it stopped early).
    pub fn run(
        vocab: &MotifVocab,
        budget: Option<&Formula>,
        tokens: &[u32],
    ) -> std::result::Result<Self, usize> {
        let mut machine = Self::new();
        for (i, &token) in tokens.iter().enumerate() {
            if machine.apply(vocab, budget, token).is_err() {
                return Err(i);
            }
        }
        if machine.done() {
            Ok(machine)
        } else {
            Err(tokens.len())
        }
    }
}

/// The text formats of the Lean reference executable (`lean/MotifDecoder`,
/// `motifcheck`), so this machine can be run on the same files and its output
/// compared line by line.
pub mod text {
    use super::*;

    /// A token id no phase allows: what an unreadable word becomes.
    pub const REJECTED: u32 = u32::MAX;

    /// A vocabulary file: `N`, then per motif
    /// `n e_0.. f_0.. k i_1 j_1 o_1 ..` (atomic numbers, free valences,
    /// bonds).
    pub fn vocab(text: &str) -> Result<MotifVocab> {
        let bad = |what: &str| Error::config(format!("motif vocab text: {what}"));
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let count: usize = lines
            .next()
            .and_then(|line| line.trim().parse().ok())
            .ok_or_else(|| bad("missing motif count"))?;
        let mut parts = Vec::with_capacity(count);
        for line in lines.by_ref().take(count) {
            let numbers: Vec<u8> = line
                .split_whitespace()
                .map(|w| w.parse().map_err(|_| bad("a word is not a small number")))
                .collect::<Result<_>>()?;
            let n = usize::from(*numbers.first().ok_or_else(|| bad("empty motif line"))?);
            let k = usize::from(*numbers.get(1 + 2 * n).ok_or_else(|| bad("short motif line"))?);
            if numbers.len() != 2 + 2 * n + 3 * k {
                return Err(bad("a motif line has the wrong length"));
            }
            parts.push((
                numbers[1..1 + n].to_vec(),
                numbers[1 + n..1 + 2 * n].to_vec(),
                numbers[2 + 2 * n..].chunks(3).map(|b| (b[0], b[1], b[2])).collect(),
            ));
        }
        if parts.len() != count || lines.next().is_some() {
            return Err(bad("the number of motif lines is not the count"));
        }
        MotifVocab::from_parts(parts)
    }

    /// One word of a sequence line: `M<id>`, `A<index>`, `B<order>` or `E`.
    /// Anything else, and any number outside the id space, is [`REJECTED`].
    pub fn token(word: &str) -> u32 {
        let number = |text: &str| text.parse::<u32>().ok();
        match word.split_at_checked(1) {
            Some(("E", "")) => END,
            Some(("M", rest)) => number(rest)
                .and_then(|m| m.checked_add(MOTIF_BASE))
                .unwrap_or(REJECTED),
            Some(("A", rest)) => number(rest)
                .filter(|&a| (a as usize) < MAX_MOTIF_ATOMS)
                .map_or(REJECTED, |a| ATOM_BASE + a),
            Some(("B", rest)) => number(rest)
                .filter(|o| (1..=3).contains(o))
                .map_or(REJECTED, |o| BOND_BASE + o),
            _ => REJECTED,
        }
    }

    /// One line of a budgets file: `-` or an empty line for none, else
    /// `H e_1 c_1 ..` (hydrogens, then atomic number and count pairs; the
    /// first entry for an element wins; an element without a count is an
    /// error).
    pub fn budget(line: &str) -> Result<Option<Formula>> {
        if line.trim() == "-" || line.trim().is_empty() {
            return Ok(None);
        }
        let bad = |what: &str| Error::config(format!("motif budget text: {what}"));
        let numbers: Vec<u16> = line
            .split_whitespace()
            .map(|w| w.parse().map_err(|_| bad("a word is not a number")))
            .collect::<Result<_>>()?;
        let (&hydrogens, pairs) = numbers.split_first().ok_or_else(|| bad("empty line"))?;
        if pairs.len() % 2 != 0 {
            return Err(bad("an element without a count"));
        }
        let mut formula = [0u16; 10];
        formula[HYDROGEN] = hydrogens;
        let mut seen = [false; 10];
        for pair in pairs.chunks(2) {
            let e = ELEMENTS
                .iter()
                .position(|&z| u16::from(z) == pair[0])
                .filter(|&e| e != HYDROGEN)
                .ok_or_else(|| bad("an element outside the alphabet"))?;
            if !seen[e] {
                seen[e] = true;
                formula[e] = pair[1];
            }
        }
        Ok(Some(formula))
    }

    /// The executable's output line for one sequence:
    /// `OK <atoms> <bonds> | <elements> | <free> | <bonds>` or
    /// `ERR <index of the first rejected token>`.
    pub fn check(vocab: &MotifVocab, budget: Option<&Formula>, line: &str) -> String {
        let tokens: Vec<u32> = line.split_whitespace().map(token).collect();
        match MotifMachine::run(vocab, budget, &tokens) {
            Err(index) => format!("ERR {index}"),
            Ok(machine) => {
                let join = |values: Vec<String>| values.join(" ");
                format!(
                    "OK {} {} | {} | {} | {}",
                    machine.elements.len(),
                    machine.bonds.len(),
                    join(machine
                        .elements
                        .iter()
                        .map(|&e| ELEMENTS[usize::from(e)].to_string())
                        .collect()),
                    join(machine.free.iter().map(u8::to_string).collect()),
                    join(machine
                        .bonds
                        .iter()
                        .map(|(i, j, o)| format!("{i} {j} {o}"))
                        .collect()),
                )
            }
        }
    }
}

/// Token ids of the sequence model. The output side (what the decoder may
/// emit) comes first, so a logit row can be cut at [`n_out`](Self::n_out).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Output ids: padding, `END`, bonds, atoms, motifs.
    pub n_out: u32,
    /// Start of a prefix.
    pub bos: u32,
    /// End of a prefix: the root motif is predicted from here.
    pub sep: u32,
    /// Separator between fingerprint confidence levels.
    pub level: u32,
    /// `count_base + e * (MAX_COUNT + 1) + c` says element `e` of
    /// [`ELEMENTS`] occurs `c` times.
    pub count_base: u32,
    /// `fingerprint_base + bit` is a fingerprint bit.
    pub fingerprint_base: u32,
    /// Size of the whole id space.
    pub vocab_size: u32,
}

impl Layout {
    /// The layout for a vocabulary of `motifs` motifs.
    pub fn new(motifs: usize) -> Self {
        let n_out = MOTIF_BASE + motifs as u32;
        let count_base = n_out + 3;
        let fingerprint_base = count_base + 10 * (u32::from(MAX_COUNT) + 1);
        Self {
            n_out,
            bos: n_out,
            sep: n_out + 1,
            level: n_out + 2,
            count_base,
            fingerprint_base,
            vocab_size: fingerprint_base + 4096,
        }
    }

    /// The conditioning prefix of a query: `BOS`, the formula as one count
    /// token per element, the fingerprint bits in three confidence
    /// levels (probability above 0.5, above 0.25, the rest; each level in bit
    /// order, separated by the level token), `SEP`. At most
    /// [`FINGERPRINT_TOKENS`] bits are kept, the most probable first (ties
    /// by lower bit). Counts above [`MAX_COUNT`] are clamped. The result is
    /// at most [`PREFIX_LEN`] ids.
    pub fn prefix(&self, formula: &Formula, fingerprint: &SparseFingerprint) -> Vec<u32> {
        let mut out = Vec::with_capacity(PREFIX_LEN);
        out.push(self.bos);
        for (e, &count) in formula.iter().enumerate() {
            out.push(
                self.count_base
                    + e as u32 * (u32::from(MAX_COUNT) + 1)
                    + u32::from(count.min(MAX_COUNT)),
            );
        }
        let mut kept: Vec<(u16, f32)> = fingerprint.entries.clone();
        kept.sort_by(|x, y| y.1.total_cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
        kept.truncate(FINGERPRINT_TOKENS);
        let level_of = |p: f32| {
            if p > 0.5 {
                0
            } else if p > 0.25 {
                1
            } else {
                2
            }
        };
        for level in 0..3 {
            let mut bits: Vec<u16> = kept
                .iter()
                .filter(|entry| level_of(entry.1) == level)
                .map(|entry| entry.0)
                .collect();
            bits.sort_unstable();
            out.extend(bits.into_iter().map(|bit| self.fingerprint_base + u32::from(bit)));
            if level < 2 {
                out.push(self.level);
            }
        }
        out.push(self.sep);
        out
    }
}

/// What [`beam_search`] returns.
#[derive(Clone, Debug, Default)]
pub struct BeamOutput {
    /// Finished sequences with their log-probability under the masked
    /// distribution, best first.
    pub finished: Vec<(Vec<u32>, f64)>,
    /// Decoder rows stepped.
    pub row_steps: usize,
}

/// One candidate continuation: the best score first, ties by the lower
/// parent row and then the lower token, so a search is deterministic.
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    score: f64,
    row: usize,
    token: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.row.cmp(&self.row))
            .then_with(|| other.token.cmp(&self.token))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Beam search over motif sequences with [`MotifMachine::allowed`] as the
/// mask, for one query and one formula.
///
/// `first` holds the logits of the root position (`n_out` values).
/// `advance(parents, tokens)` must continue row `parents[i]` of the decoder
/// with `tokens[i]` and return the next logits, `n_out` per row in that
/// order.
///
/// A hypothesis's score is the sum of its tokens' log-probabilities, each
/// normalised over the tokens the mask allows at that point. Every step
/// takes the continuations of all live hypotheses in score order: one that
/// completes the sequence is recorded, one the budget can no longer complete
/// ([`MotifMachine::dead`]) is skipped, and the first `width` others are
/// kept. Skipped and finished continuations never use up a slot, so a valid
/// completion is not lost to a better-scoring dead end from the same parent.
/// A finished sequence is recorded only if it is reached before the step's
/// `width` live continuations are found; this is a beam, not an exhaustive
/// search, and a completion behind that cut is not returned. A logit of
/// negative infinity is a continuation of probability zero and is never
/// taken; a row with a NaN or positive-infinite allowed logit, or with no
/// finite one, is dropped. The search stops when no hypothesis is live or
/// after `max_tokens` tokens: a sequence of exactly `max_tokens` tokens can
/// be returned, and `advance` is called at most `max_tokens - 1` times.
/// `n_out` must cover every output id of `vocab`.
///
/// Memory: one candidate (24 bytes) per allowed token per live row, so about
/// `width * vocab.len()` candidates in a motif step.
pub fn beam_search<F>(
    vocab: &MotifVocab,
    budget: &Formula,
    n_out: usize,
    first: Vec<f32>,
    width: usize,
    max_tokens: usize,
    mut advance: F,
) -> Result<BeamOutput>
where
    F: FnMut(&[u32], &[u32]) -> Result<Vec<f32>>,
{
    if width == 0 || first.len() != n_out || n_out < MOTIF_BASE as usize + vocab.len() {
        return Err(Error::config(format!(
            "beam_search: width {width}, {} first logits, {n_out} output ids for {} motifs",
            first.len(),
            vocab.len()
        )));
    }
    let mut logits = first;
    let mut live = vec![(MotifMachine::new(), Vec::<u32>::new(), 0.0f64)];
    let mut out = BeamOutput::default();
    let mut allowed = Vec::new();
    for step in 0..max_tokens {
        let mut candidates: Vec<Candidate> = Vec::new();
        for (row, (machine, _, score)) in live.iter().enumerate() {
            machine.allowed_tokens(vocab, Some(budget), &mut allowed);
            let scores = &logits[row * n_out..(row + 1) * n_out];
            let top = allowed
                .iter()
                .map(|&t| scores[t as usize])
                .fold(f32::NEG_INFINITY, f32::max);
            let norm = f64::from(top)
                + allowed
                    .iter()
                    .map(|&t| f64::from(scores[t as usize] - top).exp())
                    .sum::<f64>()
                    .ln();
            let unusable = |&t: &u32| {
                let logit = scores[t as usize];
                logit.is_nan() || logit == f32::INFINITY
            };
            if allowed.is_empty() || !norm.is_finite() || allowed.iter().any(unusable) {
                continue;
            }
            candidates.extend(
                allowed
                    .iter()
                    .filter(|&&token| scores[token as usize].is_finite())
                    .map(|&token| Candidate {
                        score: score + f64::from(scores[token as usize]) - norm,
                        row,
                        token,
                    }),
            );
        }
        let mut heap = std::collections::BinaryHeap::from(candidates);
        let mut next = Vec::with_capacity(width);
        let mut parents: Vec<u32> = Vec::with_capacity(width);
        let mut fed: Vec<u32> = Vec::with_capacity(width);
        while next.len() < width {
            let Some(candidate) = heap.pop() else { break };
            if !candidate.score.is_finite() {
                continue;
            }
            let (machine, tokens, _) = &live[candidate.row];
            let mut machine = machine.clone();
            machine.apply(vocab, Some(budget), candidate.token)?;
            if machine.dead(budget) {
                continue;
            }
            let mut tokens = tokens.clone();
            tokens.push(candidate.token);
            if machine.done() {
                out.finished.push((tokens, candidate.score));
                continue;
            }
            next.push((machine, tokens, candidate.score));
            parents.push(candidate.row as u32);
            fed.push(candidate.token);
        }
        // The last token a sequence may hold has been chosen: nothing would
        // read the logits of another step.
        if next.is_empty() || step + 1 == max_tokens {
            break;
        }
        logits = advance(&parents, &fed)?;
        if logits.len() != fed.len() * n_out {
            return Err(Error::shape(format!(
                "beam_search: advance returned {} logits for {} rows of {n_out}",
                logits.len(),
                fed.len()
            )));
        }
        out.row_steps += fed.len();
        live = next;
    }
    out.finished.sort_by(|x, y| y.1.total_cmp(&x.1));
    Ok(out)
}

fn gather_leading<R: Runtime, E: FloatElem>(
    value: &Tensor<R, E>,
    parents: &IdTensor<R>,
    rows: usize,
) -> Result<Tensor<R, E>> {
    let shape = value.shape().clone();
    if shape.rank() == 0 || shape.dim(0) != rows {
        return Err(Error::shape(format!(
            "gather_lm_cache: a buffer of shape {shape} does not start with {rows} rows"
        )));
    }
    let width = shape.num_elements() / rows;
    let gathered = gather_rows(&value.reshape(vec![rows, width])?, parents)?;
    let mut dims = shape.dims().to_vec();
    dims[0] = parents.len();
    gathered.reshape(Shape::new(dims))
}

/// Reorder the rows of a sequence model's decoding cache: row `i` of the
/// result is row `parents[i]` of `cache`. This is the beam step (a kept
/// hypothesis continues from its parent's state). Every layer must be a
/// Mamba layer; the result is detached. A parent at or past `rows` is
/// [`Error::Shape`].
pub fn gather_lm_cache<R: Runtime, E: FloatElem>(
    cache: &[LayerCache<R, E>],
    parents: &IdTensor<R>,
    rows: usize,
) -> Result<Vec<LayerCache<R, E>>> {
    // One read of the parent ids, so a parent past the rows is an error
    // here instead of an out-of-bounds read inside the gather.
    check_gather_ids(parents, rows)?;
    let take = |value: &Var<R, E>| -> Result<Var<R, E>> {
        Ok(Var::constant(gather_leading(value.tensor(), parents, rows)?))
    };
    let mut out = Vec::with_capacity(cache.len());
    for (l, layer) in cache.iter().enumerate() {
        let mamba = layer.mamba.as_ref().ok_or_else(|| {
            Error::config(format!("gather_lm_cache: layer {l} is not a Mamba layer"))
        })?;
        out.push(LayerCache {
            mamba: Some(MixerCache {
                ssm: SsmState {
                    h: take(&mamba.ssm.h)?,
                    last_u: take(&mamba.ssm.last_u)?,
                    angle: match &mamba.ssm.angle {
                        Some(angle) => Some(take(angle)?),
                        None => None,
                    },
                },
                conv: match &mamba.conv {
                    Some(conv) => Some(take(conv)?),
                    None => None,
                },
            }),
            attention: AttentionCache::default(),
        });
    }
    Ok(out)
}
