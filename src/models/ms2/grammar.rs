//! Graph-action grammar and canonical traversal (`grammar-bfs-v1`,
//! `bfs-canon-v1`; contract §§4.4–4.5 and 7.4).
//!
//! [`TraceState`] replays token prefixes against the legality rules and
//! reports the legality masks the decoder conditions on;
//! [`canonical_trace`] is the lexicographically smallest token sequence over
//! all breadth-first traversals, found by depth-first branch-and-bound.

use crate::error::{Error, Result};

use super::chem::{self, Composition};
use super::graph::MolGraph;

/// Grammar version of contract §4.4.
pub const GRAMMAR_VERSION: &str = "grammar-bfs-v1";
/// Traversal version of contract §7.4.
pub const TRAVERSAL_VERSION: &str = "bfs-canon-v1";

/// Padding kind: never legal inside a trace.
pub const PAD: u8 = 0;
/// First token of every trace.
pub const START: u8 = 1;
/// Add an atom: the root, a child, then its ring closures.
pub const ADD_ATOM: u8 = 2;
/// Close a ring between the newest atom and an earlier one.
pub const CLOSE_RING: u8 = 3;
/// Final token; nothing is legal after it.
pub const STOP: u8 = 4;

/// One grammar token: `(kind, atom_type, bond_order, pointer)`.
///
/// The derived field-ordered `Ord` is the canonical token comparison of
/// contract §7.4: sequences compare element-wise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token {
    /// Token kind ([`PAD`], [`START`], [`ADD_ATOM`], [`CLOSE_RING`], [`STOP`]).
    pub kind: u8,
    /// Atom type id for [`ADD_ATOM`], else 0.
    pub atom_type: u8,
    /// Bond order for non-root [`ADD_ATOM`] and [`CLOSE_RING`], else 0.
    pub bond: u8,
    /// Parent or closure target for non-root [`ADD_ATOM`] and
    /// [`CLOSE_RING`], else 0.
    pub pointer: u8,
}

/// Trace size limits; [`Limits::V0`] is the frozen V0 pair.
///
/// The fields are private: [`Limits::new`] enforces `1 <= max_atoms <= 32`
/// and `max_closures <= 32`, since atom pointers are stored in `u8` fields
/// and the legality masks are 32-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    max_atoms: usize,
    max_closures: usize,
}

impl Limits {
    /// The V0 limits: 16 atoms, 4 closures.
    pub const V0: Limits = Limits {
        max_atoms: 16,
        max_closures: 4,
    };

    /// Validated limits: `1 <= max_atoms <= 32` (a trace needs at least the
    /// root, and pointers address atoms in 32-bit masks) and
    /// `max_closures <= 32` (closure pointers share the same masks).
    pub fn new(max_atoms: usize, max_closures: usize) -> crate::error::Result<Self> {
        if !(1..=32).contains(&max_atoms) {
            return Err(crate::error::Error::config(format!(
                "Limits::new: max_atoms {max_atoms} is not in 1..=32"
            )));
        }
        if max_closures > 32 {
            return Err(crate::error::Error::config(format!(
                "Limits::new: max_closures {max_closures} exceeds 32"
            )));
        }
        Ok(Self {
            max_atoms,
            max_closures,
        })
    }

    /// Maximum atoms per trace.
    pub fn max_atoms(&self) -> usize {
        self.max_atoms
    }

    /// Maximum ring closures per trace.
    pub fn max_closures(&self) -> usize {
        self.max_closures
    }

    /// Longest possible trace: `2 + max_atoms + max_closures`.
    pub fn max_steps(&self) -> usize {
        2 + self.max_atoms + self.max_closures
    }
}

/// Per-field legality masks at one step: bit `i` set means value `i` is legal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegalMasks {
    /// Legal token kinds.
    pub kinds: u32,
    /// Legal atom types (conditioned on the taken kind).
    pub atom_types: u32,
    /// Legal bond orders (conditioned on kind and atom type).
    pub bonds: u32,
    /// Legal pointers (conditioned on kind, atom type and bond).
    pub pointers: u32,
}

/// Grammar state of a trace prefix: the legality rules of contract §4.4.
pub struct TraceState {
    limits: Limits,
    budget: Option<Composition>,
    types: Vec<u8>,
    residual: Vec<u8>,
    parent_of: Vec<Option<usize>>,
    last_parent: usize,
    last_close: Option<usize>,
    closures: usize,
    used: Composition,
    bonded: Vec<(usize, usize)>,
    bonds: Vec<(usize, usize, u8)>,
    step: usize,
    stopped: bool,
}

impl TraceState {
    /// Empty grammar state under these limits and formula budget (`None`
    /// disables the budget rule; the budget counts elements and hydrogens).
    pub fn new(limits: Limits, budget: Option<Composition>) -> Self {
        Self {
            limits,
            budget,
            types: Vec::new(),
            residual: Vec::new(),
            parent_of: Vec::new(),
            last_parent: 0,
            last_close: None,
            closures: 0,
            used: [0; 10],
            bonded: Vec::new(),
            bonds: Vec::new(),
            step: 0,
            stopped: false,
        }
    }

    /// Whether an atom type fits the remaining formula budget.
    fn type_fits(&self, id: u8) -> bool {
        let Some(budget) = &self.budget else {
            return true;
        };
        let Some(t) = chem::atom_type(id) else {
            return false;
        };
        self.used[t.element] < budget[t.element]
            && self.used[chem::HYDROGEN] + u16::from(t.hydrogens) <= budget[chem::HYDROGEN]
    }

    /// Atom types the root may take: every type the budget still allows.
    fn root_types(&self) -> u32 {
        let mut types = 0u32;
        for id in 1..=17u8 {
            if self.type_fits(id) {
                types |= 1u32 << id;
            }
        }
        types
    }

    /// Pointer bitmask of existing atoms that can take one more bond of
    /// order `bond` from a non-root child (rule 2: at or after the last
    /// parent; rule 5: residual valence).
    fn add_pointers(&self, bond: u8) -> u32 {
        if self.types.is_empty() || self.types.len() >= self.limits.max_atoms {
            return 0;
        }
        let mut mask = 0u32;
        for p in self.last_parent..self.types.len() {
            if self.residual[p] >= bond {
                mask |= 1u32 << p;
            }
        }
        mask
    }

    /// Whether any non-root child can be added at all.
    fn has_add(&self) -> bool {
        if self.types.is_empty() || self.types.len() >= self.limits.max_atoms {
            return false;
        }
        for id in 1..=17u8 {
            let Some(t) = chem::atom_type(id) else {
                continue;
            };
            if !self.type_fits(id) {
                continue;
            }
            for bond in 1..=3u8 {
                if bond > t.valence - t.hydrogens {
                    continue;
                }
                if self.add_pointers(bond) != 0 {
                    return true;
                }
            }
        }
        false
    }

    /// Bond order of the newest atom to atom `p`, if that ring closure is
    /// already present.
    fn is_bonded(&self, p: usize, newest: usize) -> bool {
        self.bonded.contains(&(p, newest))
    }

    /// Pointer bitmask for a ring closure of order `bond` on the newest atom
    /// (rule 3: above the newest atom's parent and the previous closure of
    /// the same atom; rule 5: residual valence on both ends, no double bond).
    fn close_pointers(&self, bond: u8) -> u32 {
        if self.types.len() < 2 || self.closures >= self.limits.max_closures {
            return 0;
        }
        let newest = self.types.len() - 1;
        if self.residual[newest] < bond {
            return 0;
        }
        let parent_plus_one = self.parent_of[newest].map_or(0, |p| p + 1);
        let close_plus_one = self.last_close.map_or(0, |p| p + 1);
        let lo = parent_plus_one.max(close_plus_one);
        let mut mask = 0u32;
        for p in lo..newest {
            if self.residual[p] >= bond && !self.is_bonded(p, newest) {
                mask |= 1u32 << p;
            }
        }
        mask
    }

    /// Whether any ring closure is available on the newest atom.
    fn has_close(&self) -> bool {
        if self.types.len() < 2 || self.closures >= self.limits.max_closures {
            return false;
        }
        (1..=3u8).any(|bond| self.close_pointers(bond) != 0)
    }

    /// The kind mask at this step: STOP once an atom exists (rule 8), plus
    /// ADD_ATOM / CLOSE_RING when some completion of their fields is legal
    /// (a kind is legal only then).
    fn kind_mask(&self) -> u32 {
        if self.stopped {
            return 0;
        }
        if self.step == 0 {
            return 1u32 << START;
        }
        if self.step == 1 {
            // The root is legal only if some atom type fits the budget.
            return if self.root_types() != 0 {
                1u32 << ADD_ATOM
            } else {
                0
            };
        }
        let mut kinds = 1u32 << STOP;
        if self.has_add() {
            kinds |= 1u32 << ADD_ATOM;
        }
        if self.has_close() {
            kinds |= 1u32 << CLOSE_RING;
        }
        kinds
    }

    /// Legality masks at this step, conditioned on the earlier fields of
    /// `taken`: atom types on its kind, bonds on kind and atom type,
    /// pointers on kind, atom type and bond. Fields the taken kind does not
    /// use come back as 0.
    pub fn masks(&self, taken: Token) -> LegalMasks {
        if self.stopped {
            return LegalMasks {
                kinds: 0,
                atom_types: 0,
                bonds: 0,
                pointers: 0,
            };
        }
        if self.step == 0 {
            return LegalMasks {
                kinds: 1u32 << START,
                atom_types: 0,
                bonds: 0,
                pointers: 0,
            };
        }
        if self.step == 1 {
            return LegalMasks {
                kinds: self.kind_mask(),
                atom_types: self.root_types(),
                bonds: 0,
                pointers: 0,
            };
        }
        let kinds = self.kind_mask();
        if taken.kind == ADD_ATOM && self.has_add() {
            let mut types = 0u32;
            let mut bonds = 0u32;
            let mut pointers = 0u32;
            for id in 1..=17u8 {
                let Some(t) = chem::atom_type(id) else {
                    continue;
                };
                if !self.type_fits(id) {
                    continue;
                }
                let mut type_ok = false;
                for bond in 1..=3u8 {
                    if bond > t.valence - t.hydrogens {
                        continue;
                    }
                    let ptrs = self.add_pointers(bond);
                    if ptrs == 0 {
                        continue;
                    }
                    type_ok = true;
                    if id == taken.atom_type {
                        bonds |= 1u32 << bond;
                        if bond == taken.bond {
                            pointers |= ptrs;
                        }
                    }
                }
                if type_ok {
                    types |= 1u32 << id;
                }
            }
            return LegalMasks {
                kinds,
                atom_types: types,
                bonds,
                pointers,
            };
        }
        if taken.kind == CLOSE_RING && self.has_close() {
            let mut bonds = 0u32;
            let mut pointers = 0u32;
            for bond in 1..=3u8 {
                let ptrs = self.close_pointers(bond);
                if ptrs != 0 {
                    bonds |= 1u32 << bond;
                    if bond == taken.bond {
                        pointers |= ptrs;
                    }
                }
            }
            return LegalMasks {
                kinds,
                atom_types: 0,
                bonds,
                pointers,
            };
        }
        LegalMasks {
            kinds,
            atom_types: 0,
            bonds: 0,
            pointers: 0,
        }
    }

    /// Whether `token` itself is legal here, including the rule that fields
    /// the kind does not use must be zero.
    pub fn is_legal(&self, token: Token) -> bool {
        if self.stopped {
            return false;
        }
        if self.step == 0 {
            return token
                == (Token {
                    kind: START,
                    atom_type: 0,
                    bond: 0,
                    pointer: 0,
                });
        }
        if self.step == 1 {
            return token.kind == ADD_ATOM
                && chem::atom_type(token.atom_type).is_some()
                && self.type_fits(token.atom_type)
                && token.bond == 0
                && token.pointer == 0;
        }
        match token.kind {
            ADD_ATOM => {
                let Some(t) = chem::atom_type(token.atom_type) else {
                    return false;
                };
                if self.types.len() >= self.limits.max_atoms
                    || !(1..=3u8).contains(&token.bond)
                    || token.bond > t.valence - t.hydrogens
                    || !self.type_fits(token.atom_type)
                {
                    return false;
                }
                let p = usize::from(token.pointer);
                p >= self.last_parent && p < self.types.len() && self.residual[p] >= token.bond
            }
            CLOSE_RING => {
                if token.atom_type != 0 || !(1..=3u8).contains(&token.bond) {
                    return false;
                }
                let p = usize::from(token.pointer);
                let newest = self.types.len() - 1;
                let parent_plus_one = self.parent_of[newest].map_or(0, |q| q + 1);
                let close_plus_one = self.last_close.map_or(0, |q| q + 1);
                p >= parent_plus_one.max(close_plus_one)
                    && p < newest
                    && self.residual[p] >= token.bond
                    && self.residual[newest] >= token.bond
                    && !self.is_bonded(p, newest)
                    && self.closures < self.limits.max_closures
            }
            STOP => token.atom_type == 0 && token.bond == 0 && token.pointer == 0,
            _ => false,
        }
    }

    /// Apply a legal token; illegal tokens are an error naming the step
    /// index and the token.
    pub fn apply(&mut self, token: Token) -> Result<()> {
        if !self.is_legal(token) {
            return Err(Error::config(format!(
                "illegal token at step {}: kind={} atom_type={} bond={} pointer={}",
                self.step, token.kind, token.atom_type, token.bond, token.pointer
            )));
        }
        match token.kind {
            START => {
                // Step 0 only (checked by legality): no graph state yet.
            }
            ADD_ATOM => {
                let t = chem::atom_type(token.atom_type).expect("legality checked the type");
                self.used[t.element] += 1;
                self.used[chem::HYDROGEN] += u16::from(t.hydrogens);
                self.types.push(token.atom_type);
                self.residual.push(t.valence - t.hydrogens);
                if self.step == 1 {
                    self.parent_of.push(None);
                } else {
                    let p = usize::from(token.pointer);
                    let newest = self.types.len() - 1;
                    self.residual[p] -= token.bond;
                    self.residual[newest] -= token.bond;
                    self.parent_of.push(Some(p));
                    self.bonded.push((p, newest));
                    self.bonds.push((p, newest, token.bond));
                    self.last_parent = p;
                }
                self.last_close = None;
            }
            CLOSE_RING => {
                let p = usize::from(token.pointer);
                let newest = self.types.len() - 1;
                self.residual[p] -= token.bond;
                self.residual[newest] -= token.bond;
                self.bonded.push((p, newest));
                self.bonds.push((p, newest, token.bond));
                self.last_close = Some(p);
                self.closures += 1;
            }
            STOP => {
                self.stopped = true;
            }
            _ => unreachable!("legality rules out other kinds"),
        }
        self.step += 1;
        Ok(())
    }

    /// Tokens applied so far.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Atoms added so far.
    pub fn atoms(&self) -> usize {
        self.types.len()
    }

    /// Whether STOP was applied.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Residual valence per atom in trace order; at STOP this is the open
    /// attachment valence of contract §4.5.
    pub fn residual_valence(&self) -> &[u8] {
        &self.residual
    }

    /// Whether any legal action exists (STOP counts: a prefix with no other
    /// legal action emits STOP).
    pub fn has_legal_action(&self) -> bool {
        if self.stopped {
            return false;
        }
        if self.step == 0 {
            return true;
        }
        self.kind_mask() != 0
    }

    /// The graph built so far (atom `i` is the `i`-th added atom).
    pub fn graph(&self) -> Result<MolGraph> {
        MolGraph::new(self.types.clone(), self.bonds.clone())
    }
}

/// Replay a trace under these limits and budget, returning the end state.
///
/// Errors name the first illegal step index and its token.
pub fn replay(trace: &[Token], limits: Limits, budget: Option<Composition>) -> Result<TraceState> {
    let mut state = TraceState::new(limits, budget);
    for token in trace {
        state.apply(*token)?;
    }
    Ok(state)
}

/// Index of the first token the grammar forbids, or `None` for a legal trace.
pub fn first_illegal_step(
    trace: &[Token],
    limits: Limits,
    budget: Option<Composition>,
) -> Option<usize> {
    let mut state = TraceState::new(limits, budget);
    for (i, token) in trace.iter().enumerate() {
        if !state.is_legal(*token) {
            return Some(i);
        }
        if state.apply(*token).is_err() {
            return Some(i);
        }
    }
    None
}

/// A canonical trace: the token sequence, the placement order and the cost.
#[derive(Debug, Clone)]
pub struct Canonical {
    /// The lexicographically smallest token sequence.
    pub trace: Vec<Token>,
    /// `order[i]` is the graph atom placed `i`-th.
    pub order: Vec<usize>,
    /// Candidate extensions explored.
    pub expansions: usize,
}

/// Work limit of contract §7.4.
pub const CANONICAL_WORK_LIMIT: usize = 200_000;

/// The lexicographically smallest token sequence over all breadth-first
/// traversals of a connected graph (contract §7.4).
///
/// Every root is tried, and at the queue head every order of discovering the
/// head's undiscovered neighbours. Each new atom emits its ADD_ATOM then one
/// CLOSE_RING per bond to an earlier atom other than its parent, in
/// increasing pointer order; the trace starts with START and a root ADD_ATOM
/// and ends with STOP.
///
/// The search is depth-first branch-and-bound: the best complete trace is
/// kept and a partial trace is pruned as soon as it is lexicographically
/// greater than the best prefix of the same length (equal prefixes continue).
/// Each candidate extension counts one expansion; past `max_expansions` the
/// search stops with a `canonicalization_budget_exceeded` error.
pub fn canonical_trace(
    graph: &MolGraph,
    limits: Limits,
    max_expansions: usize,
) -> Result<Canonical> {
    let n = graph.atoms().len();
    if n == 0 {
        return Err(Error::config("canonical_trace: empty graph"));
    }
    if n > limits.max_atoms {
        return Err(Error::config(format!(
            "canonical_trace: {n} atoms exceed max_atoms {}",
            limits.max_atoms
        )));
    }
    if !graph.is_connected() {
        return Err(Error::config("canonical_trace: disconnected graph"));
    }
    if graph.ring_closures() > limits.max_closures {
        return Err(Error::config(format!(
            "canonical_trace: {} ring closures exceed max_closures {}",
            graph.ring_closures(),
            limits.max_closures
        )));
    }
    let types = graph.atoms().to_vec();
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for (a, b, order) in graph.bonds() {
        adj[*a].push((*b, *order));
        adj[*b].push((*a, *order));
    }
    let bond_order = |u: usize, v: usize| -> u8 {
        adj[u]
            .iter()
            .find(|(x, _)| *x == v)
            .map(|(_, o)| *o)
            .expect("bond exists")
    };

    let mut search = Search {
        types: &types,
        adj: &adj,
        bond_order: &bond_order,
        n,
        max_expansions,
        expansions: 0,
        best_trace: None,
        best_order: Vec::new(),
        over_budget: false,
    };
    // Smaller root types first: the second token already orders traces, so
    // the first complete trace prunes well.
    let mut roots: Vec<usize> = (0..n).collect();
    roots.sort_by_key(|r| types[*r]);
    for root in roots {
        search.expansions += 1;
        if search.expansions > search.max_expansions {
            search.over_budget = true;
            break;
        }
        let mut order = vec![root];
        let mut placed = vec![false; n];
        placed[root] = true;
        let mut tokens = vec![
            Token {
                kind: START,
                atom_type: 0,
                bond: 0,
                pointer: 0,
            },
            Token {
                kind: ADD_ATOM,
                atom_type: types[root],
                bond: 0,
                pointer: 0,
            },
        ];
        if search.pruned(&tokens) {
            continue;
        }
        search.dfs(&mut order, &mut placed, 0, &mut tokens);
        if search.over_budget {
            break;
        }
    }
    if search.over_budget {
        return Err(Error::Unsupported(format!(
            "canonicalization_budget_exceeded: {} expansions over limit {max_expansions}",
            search.expansions
        )));
    }
    let trace = search
        .best_trace
        .ok_or_else(|| Error::config("canonical_trace: no breadth-first traversal found"))?;
    debug_assert!(replay(&trace, limits, None).is_ok());
    Ok(Canonical {
        trace,
        order: search.best_order,
        expansions: search.expansions,
    })
}

/// Branch-and-bound state for [`canonical_trace`].
struct Search<'a> {
    /// Atom type id per graph atom.
    types: &'a [u8],
    /// Neighbours with bond orders, per graph atom.
    adj: &'a [Vec<(usize, u8)>],
    /// Bond order lookup between bonded atoms.
    bond_order: &'a dyn Fn(usize, usize) -> u8,
    /// Atom count.
    n: usize,
    /// Expansion budget.
    max_expansions: usize,
    /// Candidate extensions explored so far.
    expansions: usize,
    /// Best complete trace (with STOP) so far.
    best_trace: Option<Vec<Token>>,
    /// Placement order behind the best trace.
    best_order: Vec<usize>,
    /// Set once the budget is spent; unwinds the search.
    over_budget: bool,
}

impl Search<'_> {
    /// Whether a partial trace already exceeds the best prefix of the same
    /// length (equal prefixes must continue).
    fn pruned(&self, tokens: &[Token]) -> bool {
        match &self.best_trace {
            None => false,
            Some(best) => tokens > &best[..tokens.len()],
        }
    }

    /// Record a complete trace when it improves on the best.
    fn complete(&mut self, order: &[usize], tokens: &[Token]) {
        let mut full = tokens.to_vec();
        full.push(Token {
            kind: STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        });
        let better = match &self.best_trace {
            None => true,
            Some(best) => full < *best,
        };
        if better {
            self.best_trace = Some(full);
            self.best_order = order.to_vec();
        }
    }

    /// Extend `order`/`tokens` from queue position `head` (which stays put:
    /// the next call advances past exhausted heads itself).
    fn dfs(
        &mut self,
        order: &mut Vec<usize>,
        placed: &mut [bool],
        head: usize,
        tokens: &mut Vec<Token>,
    ) {
        if self.over_budget {
            return;
        }
        if order.len() == self.n {
            self.complete(order, tokens);
            return;
        }
        // Advance past queue heads with no undiscovered neighbour.
        let mut h = head;
        let parent = loop {
            if h >= order.len() {
                return;
            }
            let u = order[h];
            if self.adj[u].iter().any(|(v, _)| !placed[*v]) {
                break h;
            }
            h += 1;
        };
        let u = order[parent];
        // One candidate block per undiscovered neighbour: its ADD_ATOM then
        // one CLOSE_RING per bond to a placed atom other than the parent, in
        // increasing pointer order. Smaller blocks first for early pruning.
        let mut placed_index = vec![0usize; self.n];
        for (i, a) in order.iter().enumerate() {
            placed_index[*a] = i;
        }
        let mut candidates: Vec<(usize, Vec<Token>)> = Vec::new();
        for (v, _) in &self.adj[u] {
            if placed[*v] {
                continue;
            }
            let mut block = vec![Token {
                kind: ADD_ATOM,
                atom_type: self.types[*v],
                bond: (self.bond_order)(u, *v),
                pointer: parent as u8,
            }];
            let mut closers: Vec<usize> = self.adj[*v]
                .iter()
                .filter(|(w, _)| placed[*w] && *w != u)
                .map(|(w, _)| *w)
                .collect();
            closers.sort_by_key(|w| placed_index[*w]);
            for w in closers {
                block.push(Token {
                    kind: CLOSE_RING,
                    atom_type: 0,
                    bond: (self.bond_order)(*v, w),
                    pointer: placed_index[w] as u8,
                });
            }
            candidates.push((*v, block));
        }
        candidates.sort_by(|a, b| a.1.cmp(&b.1));
        for (v, block) in candidates {
            self.expansions += 1;
            if self.expansions > self.max_expansions {
                self.over_budget = true;
                return;
            }
            tokens.extend_from_slice(&block);
            order.push(v);
            placed[v] = true;
            if !self.pruned(tokens) {
                self.dfs(order, placed, parent, tokens);
            }
            placed[v] = false;
            order.pop();
            tokens.truncate(tokens.len() - block.len());
            if self.over_budget {
                return;
            }
        }
    }
}
