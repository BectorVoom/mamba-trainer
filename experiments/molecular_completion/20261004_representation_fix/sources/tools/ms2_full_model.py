"""Complete CHNO prototype: learned substructures, Mamba-3 graph actions and ranking.

Uses the repository's PyTorch Mamba-3 reference and MS2 Python grammar reference.
The domain and traversal are explicit experiment policies, not new frozen V0 APIs.
"""
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from bench.torch_mamba3 import Block, RmsNorm
from tools.ms2.ms2_reference import (Replay, ATOM_TYPES, TYPE_ID, MASS, RESIDUAL,
                                    START, ADD, CLOSE, STOP, first_bfs_trace)

MAX_ATOMS, MAX_CLOSURES, MAX_STEPS = 32, 8, 42
ELEMENTS = ("C", "H", "N", "O")
ADDUCT_SHIFT = {"[M+H]+": 1_007_276, "[M-H]-": -1_007_276,
                "[M+Na]+": 22_989_221}
ADDUCT_IDS = {s: i + 1 for i, s in enumerate(ADDUCT_SHIFT)}
DOMAIN = "neutral-CHNO-C1to32-N0to12-O0to16-2to32atoms-8cycles-connectivity-rdkit-bfs-v1"
REPRESENTATION_VERSION = "continuous-forward-canonical-graph-v2"
# Continuous random Fourier features at two mass resolutions complement coarse
# bins without requiring a multi-million-bin spectrum prediction head.
MASS_FREQUENCIES = np.concatenate([
    np.random.default_rng(seed).normal(0, 1 / (2 * math.pi * sigma), 512)
    for seed, sigma in ((71, 0.01), (72, 0.001))
])
SPECTRUM_DIM = 2000 + 2 * len(MASS_FREQUENCIES)
CANONICAL_GRAPH_DIM = MAX_ATOMS * 21 + MAX_ATOMS * MAX_ATOMS * 5


@dataclass(frozen=True)
class Graph:
    types: tuple
    edges: tuple
    identity: str = ""
    stereo_smiles: str = ""

    @property
    def counts(self):
        counts = Counter()
        for t in self.types:
            element, hydrogens, _ = ATOM_TYPES[t]
            counts[element] += 1
            counts["H"] += hydrogens
        return tuple(counts[e] for e in ELEMENTS)

    def trace(self):
        adjacency = {i: {} for i in range(len(self.types))}
        for a, b, order in self.edges:
            adjacency[a][b] = adjacency[b][a] = order
        return first_bfs_trace(self.types, adjacency)


def identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "raw:" + smiles
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def parse_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "smiles_parse"
    # Canonical connectivity SMILES removes isotope labels. Validate the
    # original molecule before that normalization can hide an excluded label.
    excluded_label = any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    labelled = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    # Reparse canonical connectivity SMILES: a deterministic traversal independent
    # of the supplier's atom labels, without an exponential canonical BFS search.
    mol = Chem.MolFromSmiles(key)
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "disconnected"
    if not 2 <= mol.GetNumAtoms() <= MAX_ATOMS:
        return None, "atom_count"
    if mol.GetNumBonds() - mol.GetNumAtoms() + 1 > MAX_CLOSURES:
        return None, "cycle_count"
    if any(a.GetSymbol() not in ("C", "N", "O") for a in mol.GetAtoms()):
        return None, "element"
    if excluded_label:
        return None, "charged_isotopic_or_radical"
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Chem.KekulizeException:
        return None, "kekulization"
    types = []
    for a in mol.GetAtoms():
        h = a.GetTotalNumHs()
        valence = h + int(sum(b.GetBondTypeAsDouble() for b in a.GetBonds()))
        t = TYPE_ID.get((a.GetSymbol(), h, valence))
        if t is None or t > 9:
            return None, "atom_type"
        types.append(t)
    edges = tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          int(b.GetBondTypeAsDouble())) for b in mol.GetBonds()))
    graph = Graph(tuple(types), edges, key, labelled)
    c, _, n, o = graph.counts
    if c < 1 or n > 12 or o > 16:
        return None, "formula_table_domain"
    return graph, None


def canonical_candidates(smiles):
    """Deduplicate identities while validating the original candidate labels."""
    result = {}
    for raw in smiles:
        key = identity(raw)
        candidate, _ = parse_graph(raw)
        if key not in result or result[key] is None:
            result[key] = candidate
    return result


def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    # Acyclic molecules have no Murcko scaffold; connectivity groups remain intact.
    return "scaffold:" + value if value else "acyclic:" + identity(smiles)


class CompleteState(Replay):
    def __init__(self, counts):
        super().__init__(dict(zip(ELEMENTS, counts)))
        self.edges = []

    def add_options(self):
        if not self.types or len(self.types) >= MAX_ATOMS:
            return {}
        options = {}
        for t in range(1, 10):
            _, h, valence = ATOM_TYPES[t]
            if not self.type_fits(t):
                continue
            for bond in (1, 2, 3):
                if bond > valence - h:
                    continue
                pointers = [p for p in range(self.last_parent, len(self.types)) if self.residual[p] >= bond]
                if pointers:
                    options.setdefault(t, {})[bond] = pointers
        return options

    def close_options(self):
        if len(self.types) < 2 or self.closures >= MAX_CLOSURES:
            return {}
        newest = len(self.types) - 1
        low = max(self.parent_of[newest], self.last_close) + 1
        options = {}
        for bond in (1, 2, 3):
            if self.residual[newest] < bond:
                continue
            pointers = [p for p in range(low, newest) if self.residual[p] >= bond and (p, newest) not in self.bonded]
            if pointers:
                options[bond] = pointers
        return options

    def masks(self, token):
        masks, legal = super().masks(token)
        complete = self.used == Counter(self.budget) and not any(self.residual)
        if not complete:
            masks[0] &= ~(1 << STOP)
            if token[0] == STOP:
                legal = False
        return masks, legal

    def apply(self, token):
        _, legal = self.masks(token)
        if not legal:
            raise ValueError(f"illegal complete-graph action at step {self.step}: {token}")
        kind, atom, bond, pointer = token
        if kind == ADD and self.types:
            self.edges.append((pointer, len(self.types), bond))
        elif kind == CLOSE:
            self.edges.append((pointer, len(self.types) - 1, bond))
        super().apply(token)

    def snapshot(self):
        types = np.zeros(MAX_ATOMS, dtype=np.int64)
        residual = np.zeros(MAX_ATOMS, dtype=np.float32)
        adjacency = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=np.float32)
        types[:len(self.types)] = self.types
        residual[:len(self.types)] = self.residual
        for a, b, order in self.edges:
            adjacency[a, b] = adjacency[b, a] = order
        remaining = [(self.budget[e] - self.used[e]) / d for e, d in zip(ELEMENTS, (32, 66, 12, 16))]
        summary = remaining + [sum(self.residual) / 128, len(self.types) / 32,
                               self.closures / 8, self.last_parent / 32]
        return types, residual, adjacency, np.asarray(summary, dtype=np.float32)

    def graph(self):
        if not self.stopped or any(self.residual) or self.used != Counter(self.budget):
            raise ValueError("graph is incomplete")
        graph = Graph(tuple(self.types), tuple(self.edges))
        mol = Chem.RWMol()
        for t in graph.types:
            symbol, h, _ = ATOM_TYPES[t]
            atom = Chem.Atom(symbol)
            atom.SetNumExplicitHs(h)
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)
        for a, b, order in graph.edges:
            mol.AddBond(a, b, (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)[order - 1])
        try:
            Chem.SanitizeMol(mol)
        except (Chem.AtomValenceException, Chem.KekulizeException):
            return None
        key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        parsed, _ = parse_graph(key)
        return parsed if parsed is not None and parsed.counts == graph.counts else None


def motif_key(graph):
    candidates = []
    for order in itertools.permutations(range(len(graph.types))):
        pos = {old: new for new, old in enumerate(order)}
        edges = tuple(sorted((min(pos[a], pos[b]), max(pos[a], pos[b]), bnd) for a, b, bnd in graph.edges))
        candidates.append((tuple(graph.types[i] for i in order), edges))
    return min(candidates)


@lru_cache(maxsize=8192)
def motifs(graph):
    result = set()
    for t in graph.types:
        result.add(((t,), ()))
    adjacency = [[] for _ in graph.types]
    for a, b, bond in graph.edges:
        result.add(motif_key(Graph((graph.types[a], graph.types[b]), ((0, 1, bond),))))
        adjacency[a].append((b, bond))
        adjacency[b].append((a, bond))
    for center, neighbors in enumerate(adjacency):
        for (a, ba), (b, bb) in itertools.combinations(neighbors, 2):
            # Use induced bonds, including triangles; parent H counts stay fixed.
            lookup = {tuple(sorted((x, y))): o for x, y, o in graph.edges}
            edges = [(0, 1, ba), (1, 2, bb)]
            if tuple(sorted((a, b))) in lookup:
                edges.append((0, 2, lookup[tuple(sorted((a, b)))]))
            result.add(motif_key(Graph((graph.types[a], graph.types[center], graph.types[b]), tuple(edges))))
    return frozenset(result)


def vocabulary(records, cap=128):
    counts = Counter(key for r in records for key in motifs(r["graph"]))
    return sorted(counts, key=lambda k: (-counts[k], k))[:cap]


def prep_peaks(peaks, precursor):
    if not math.isfinite(precursor) or precursor <= 0:
        raise ValueError("precursor must be finite and positive")
    if any(not math.isfinite(m) or not math.isfinite(i) or i < 0 for m, i in peaks):
        raise ValueError("peaks must be finite with nonnegative intensities")
    kept = [(m, i, p) for p, (m, i) in enumerate(peaks) if 0 < m <= precursor + 2 and i > 0]
    maximum = max((i for _, i, _ in kept), default=0)
    kept = [(m, i / maximum, p) for m, i, p in kept if i / maximum >= 1e-3]
    kept = sorted(sorted(kept, key=lambda v: (-v[1], v[2]))[:160], key=lambda v: (v[0], v[2]))
    features = np.zeros((160, 7), dtype=np.float32)
    mask = np.zeros(160, dtype=np.float32)
    bins = np.zeros(2000, dtype=np.float32)
    fine = np.zeros(2 * len(MASS_FREQUENCIES), dtype=np.float64)
    for p, (m, intensity, _) in enumerate(kept):
        root = math.sqrt(intensity)
        features[p] = (m / 1000, (precursor - m) / 1000, root,
                       math.sin(2 * math.pi * m), math.cos(2 * math.pi * m),
                       math.sin(20 * math.pi * m), math.cos(20 * math.pi * m))
        mask[p] = 1
        if m < 2000:
            bins[int(m)] = max(bins[int(m)], root)
        phase = 2 * math.pi * m * MASS_FREQUENCIES
        fine += root * np.concatenate([np.cos(phase), np.sin(phase)])
    norm = np.linalg.norm(bins)
    bins = bins / norm if norm else bins
    fine_norm = np.linalg.norm(fine)
    fine = fine / fine_norm if fine_norm else fine
    target = np.concatenate([bins, fine]).astype(np.float32)
    target_norm = np.linalg.norm(target)
    return features, mask, target / target_norm if target_norm else target


def neutral_mass(row):
    if row["adduct"] not in ADDUCT_SHIFT:
        raise ValueError("unsupported adduct")
    value = Decimal(row["precursor_mz"])
    if not value.is_finite() or value <= 0:
        raise ValueError("invalid precursor")
    ion = int((value * 1_000_000).to_integral_value(rounding=ROUND_HALF_EVEN))
    resolution = Decimal(10) ** value.as_tuple().exponent
    rounding = int((resolution * 500_000).to_integral_value(rounding="ROUND_CEILING"))
    return ion - ADDUCT_SHIFT[row["adduct"]], max(1, rounding)


class FormulaTable:
    def __init__(self):
        rows = []
        for c in range(1, 33):
            for n in range(min(12, 32 - c) + 1):
                for o in range(min(16, 32 - c - n) + 1):
                    if c + n + o < 2:
                        continue
                    for h in range((2 * c + 2 + n) % 2, 2 * c + n + 3, 2):
                        counts = (c, h, n, o)
                        mass = sum(MASS[e] * number for e, number in zip(ELEMENTS, counts))
                        if mass <= 1_000_000_000:
                            rows.append((mass, counts))
        rows.sort()
        self.mass = np.asarray([r[0] for r in rows], dtype=np.int64)
        self.counts = np.asarray([r[1] for r in rows], dtype=np.int64)

    def hypotheses(self, row):
        observed, error = neutral_mass(row)
        tolerance = max(100, int(abs(observed) * 20 / 1_000_000))
        # First use a conservative superset; retain only numerical acceptance.
        lo, hi = np.searchsorted(self.mass, [observed - tolerance - error - 100,
                                            observed + tolerance + error + 100])
        candidates = []
        for mass, counts in zip(self.mass[lo:hi], self.counts[lo:hi]):
            arithmetic = math.ceil(sum(RESIDUAL[e] * int(v) for e, v in zip(ELEMENTS, counts)) / 1000) + 2
            if abs(int(mass) - observed) + error + arithmetic <= tolerance:
                candidates.append(tuple(int(v) for v in counts))
        return candidates


def canonical_graph_features(graph):
    """Full canonical connectivity and stereo labels, independent of atom order.

    Connectivity identities remain the benchmark grouping key. Stereo labels
    are retained separately so grouping cannot erase encoder input distinctions.
    Generated graphs currently carry no stereo labels.
    """
    if graph.stereo_smiles:
        mol = Chem.MolFromSmiles(graph.stereo_smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        for atom in mol.GetAtoms():
            h = atom.GetTotalNumHs()
            valence = h + int(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()))
            atom.SetIsotope(100 + TYPE_ID[(atom.GetSymbol(), h, valence)])
    else:
        editable = Chem.RWMol()
        for t in graph.types:
            element, hydrogens, _ = ATOM_TYPES[t]
            atom = Chem.Atom(element)
            atom.SetNumExplicitHs(hydrogens)
            atom.SetNoImplicit(True)
            # Internal colors retain parent atom types for incomplete motifs.
            # They are serialization labels, not chemical isotope inputs.
            atom.SetIsotope(100 + t)
            editable.AddAtom(atom)
        for a, b, order in graph.edges:
            editable.AddBond(a, b, (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)[order - 1])
        mol = editable.GetMol()
        Chem.SanitizeMol(mol)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    atoms = np.zeros((MAX_ATOMS, 21), dtype=np.float32)
    # Canonical local chiral parity and E/Z remain distinct after serialization.
    Chem.Kekulize(mol, clearAromaticFlags=True)
    edges = np.zeros((MAX_ATOMS, MAX_ATOMS, 5), dtype=np.float32)
    for atom in mol.GetAtoms():
        atoms[atom.GetIdx(), atom.GetIsotope() - 100] = 1
        chirality = int(atom.GetChiralTag())
        atoms[atom.GetIdx(), 18 + chirality] = 1
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[a, b, int(bond.GetBondTypeAsDouble()) - 1] = 1
        stereo = bond.GetStereo()
        if stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            edges[a, b, 3] = 1
        elif stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            edges[a, b, 4] = 1
        edges[b, a] = edges[a, b]
    return np.concatenate([atoms.ravel(), edges.ravel()])


def graph_tensors(graphs, device):
    length = max(len(g.types) for g in graphs)
    types = np.zeros((len(graphs), length), dtype=np.int64)
    bonds = np.zeros((len(graphs), length, length), dtype=np.float32)
    for i, g in enumerate(graphs):
        types[i, :len(g.types)] = g.types
        for a, b, order in g.edges:
            bonds[i, a, b] = bonds[i, b, a] = order
    canonical = np.stack([canonical_graph_features(g) for g in graphs])
    return (torch.as_tensor(types, device=device), torch.as_tensor(bonds, device=device),
            torch.as_tensor(canonical, device=device))


class GraphEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.atom = nn.Embedding(18, width, padding_idx=0)
        self.own = nn.ModuleList(nn.Linear(width, width) for _ in range(3))
        self.neighbor = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(3))
        self.output = nn.Linear(width * 2, width)
        self.canonical = nn.Linear(CANONICAL_GRAPH_DIM, width, bias=False)

    def forward(self, types, bonds, canonical):
        mask = (types != 0).unsqueeze(-1)
        h = self.atom(types)
        for own, neighbor in zip(self.own, self.neighbor):
            h = F.gelu(own(h) + neighbor(bonds @ h)) * mask
        summed = h.sum(1)
        return self.output(torch.cat([summed / mask.sum(1).clamp_min(1), summed / 8], -1)) + self.canonical(canonical)


def block(width):
    return Block(d_model=width, n_heads=3, head_dim=32, d_state=8, chunk=16)


class SpectrumEncoder(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.peak = nn.Linear(7, width)
        self.forward_layers = nn.ModuleList(block(width) for _ in range(2))
        self.reverse_layers = nn.ModuleList(block(width) for _ in range(2))
        self.adduct = nn.Embedding(4, 8)
        self.meta = nn.Linear(13, width)
        self.pool = nn.Linear(width * 2, width)

    def forward(self, peaks, mask, meta, adduct):
        original = self.peak(peaks) * mask.unsqueeze(-1)
        forward, reverse = original, original.flip(1)
        for layer in self.forward_layers:
            forward = layer(forward) * mask.unsqueeze(-1)
        for layer in self.reverse_layers:
            reverse = layer(reverse) * mask.flip(1).unsqueeze(-1)
        memory = (forward + reverse.flip(1)) * 0.5
        average = (memory * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
        maximum = memory.masked_fill(~mask.bool().unsqueeze(-1), -1e4).max(1).values
        maximum = torch.where(mask.sum(1, keepdim=True) > 0, maximum, torch.zeros_like(maximum))
        context = self.pool(torch.cat([average, maximum], -1))
        context = context + self.meta(torch.cat([meta, self.adduct(adduct)], -1))
        return context, memory


class SubstructurePredictor(nn.Module):
    def __init__(self, n_patterns):
        super().__init__()
        self.encoder = SpectrumEncoder()
        self.head = nn.Linear(96, n_patterns)

    def forward(self, batch):
        context, _ = self.encoder(*batch[:4])
        return self.head(context)


def batch_inputs(records, device):
    return tuple(torch.as_tensor(np.stack([r[k] for r in records]), device=device)
                 for k in ("peaks", "mask", "meta", "adduct", "bins"))


def substructure_context(model, records, device):
    graphs, weights, owners = [], [], []
    for i, r in enumerate(records):
        for pattern, confidence in r.get("predicted", []):
            graphs.append(Graph(pattern[0], pattern[1]))
            weights.append(confidence)
            owners.append(i)
    result = torch.zeros(len(records), 96, device=device)
    if graphs:
        encoded = model.graph_encoder(*graph_tensors(graphs, device))
        confidence = torch.tensor(weights, device=device).unsqueeze(-1)
        owner = torch.tensor(owners, device=device)
        result.index_add_(0, owner, encoded * confidence)
        denominator = torch.zeros(len(records), 1, device=device)
        denominator.index_add_(0, owner, confidence)
        result = result / denominator.clamp_min(1)
    return result


def teacher_batch(graphs, device):
    traces = [g.trace() for g in graphs]
    length = max(len(t) - 1 for t in traces)
    tokens = np.zeros((len(graphs), length, 4), dtype=np.int64)
    targets = np.zeros_like(tokens)
    valid = np.zeros((len(graphs), length), dtype=np.float32)
    snapshots = [np.zeros((len(graphs), length, MAX_ATOMS), dtype=np.int64),
                 np.zeros((len(graphs), length, MAX_ATOMS), dtype=np.float32),
                 np.zeros((len(graphs), length, MAX_ATOMS, MAX_ATOMS), dtype=np.float32),
                 np.zeros((len(graphs), length, 8), dtype=np.float32)]
    masks = [np.zeros((len(graphs), length, n), dtype=bool) for n in (5, 18, 4, MAX_ATOMS)]
    for b, (g, trace) in enumerate(zip(graphs, traces)):
        state = CompleteState(g.counts)
        for step, (previous, target) in enumerate(zip(trace, trace[1:])):
            state.apply(previous)
            legal_masks, legal = state.masks(target)
            if not legal:
                raise ValueError(f"teacher trace illegal: {g.identity}, step {step}, {target}")
            tokens[b, step], targets[b, step], valid[b, step] = previous, target, 1
            for out, value in zip(snapshots, state.snapshot()):
                out[b, step] = value
            for out, bitmask in zip(masks, legal_masks):
                for j in range(out.shape[-1]):
                    out[b, step, j] = bool(bitmask & (1 << j))
        state.apply(trace[-1])
        for out in masks:
            # Inactive fields and padded steps need a finite softmax too.
            empty = ~out[b].any(-1)
            out[b, empty, 0] = True
    convert = lambda a: torch.as_tensor(a, device=device)
    return convert(tokens), convert(targets), convert(valid), tuple(map(convert, snapshots)), tuple(map(convert, masks))


class CompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        width = 96
        self.spectrum_encoder = SpectrumEncoder(width)
        self.graph_encoder = GraphEncoder(width)
        self.formula = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.formula_query = nn.Linear(width, width)
        self.kind_emb = nn.Embedding(5, width, padding_idx=0)
        self.type_emb = nn.Embedding(18, width, padding_idx=0)
        self.bond_emb = nn.Embedding(4, width)
        self.pointer_emb = nn.Embedding(MAX_ATOMS, width)
        self.position = nn.Embedding(MAX_STEPS, width)
        self.state_projection = nn.Linear(8, width)
        self.residual_projection = nn.Linear(1, width)
        self.neighbor_projection = nn.Linear(width, width, bias=False)
        self.layers = nn.ModuleList(block(width) for _ in range(2))
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = RmsNorm(width)
        self.kind_head = nn.Linear(width, 5)
        self.type_head = nn.Linear(width, 18)
        self.bond_head = nn.Linear(width, 4)
        self.pointer_query = nn.Linear(width, width)
        # Fourier targets are signed; a nonnegative head would erase them.
        self.forward_spectrum = nn.Sequential(nn.Linear(width + 13, width), nn.GELU(), nn.Linear(width, SPECTRUM_DIM))

    def predict_spectrum(self, encoded, inputs):
        conditions = torch.cat([inputs[2], self.spectrum_encoder.adduct(inputs[3])], -1)
        return F.normalize(self.forward_spectrum(torch.cat([encoded, conditions], -1)), dim=-1)

    def condition(self, records, device, conditional=True):
        if not conditional:
            return torch.zeros(len(records), 96, device=device), None, None
        peaks, mask, meta, adduct, _ = batch_inputs(records, device)
        context, memory = self.spectrum_encoder(peaks, mask, meta, adduct)
        sub = substructure_context(self, records, device)
        memory = torch.cat([memory, sub.unsqueeze(1)], 1)
        padding = torch.cat([~mask.bool(), torch.zeros(len(records), 1, device=device, dtype=torch.bool)], 1)
        return context + sub, memory, padding

    def formula_embedding(self, counts):
        return self.formula(counts / counts.new_tensor([32, 66, 12, 16]))

    def hidden(self, tokens, snapshots, counts, context, memory, padding):
        types, residual, adjacency, summary = snapshots
        x = sum(embedding(tokens[..., i]) for i, embedding in enumerate((self.kind_emb, self.type_emb, self.bond_emb, self.pointer_emb)))
        x = x + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        x = x + self.state_projection(summary)
        x = x + self.formula_embedding(counts).unsqueeze(1) + context.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        if memory is not None:
            x = x + self.cross(x, memory, memory, key_padding_mask=padding, need_weights=False)[0]
        return self.norm(x)

    def fields(self, hidden, chosen_types, chosen_bonds, snapshots):
        types, residual, adjacency, _ = snapshots
        atom_memory = self.type_emb(types)
        atom_memory = atom_memory + self.neighbor_projection(adjacency @ atom_memory)
        atom_memory = atom_memory + self.residual_projection(residual.unsqueeze(-1) / 4)
        atom_memory = atom_memory + self.pointer_emb(torch.arange(MAX_ATOMS, device=hidden.device))
        selected = hidden + self.type_emb(chosen_types)
        pointer = self.pointer_query(selected + self.bond_emb(chosen_bonds))
        return self.kind_head(hidden), self.type_head(hidden), self.bond_head(selected), (pointer.unsqueeze(-2) * atom_memory).sum(-1) / math.sqrt(96)

    def likelihood(self, graphs, records, device, conditional=True):
        tokens, targets, valid, snapshots, masks = teacher_batch(graphs, device)
        counts = torch.tensor([g.counts for g in graphs], device=device, dtype=torch.float32)
        context, memory, padding = self.condition(records, device, conditional)
        hidden = self.hidden(tokens, snapshots, counts, context, memory, padding)
        logits = self.fields(hidden, targets[..., 1], targets[..., 2], snapshots)
        active = [valid, valid * (targets[..., 0] == ADD),
                  valid * ((targets[..., 0] == ADD) | (targets[..., 0] == CLOSE)) * (snapshots[0].count_nonzero(-1) > 0),
                  valid * ((targets[..., 0] == ADD) | (targets[..., 0] == CLOSE)) * (snapshots[0].count_nonzero(-1) > 0)]
        total = torch.zeros_like(valid)
        for field, (values, mask, weight) in enumerate(zip(logits, masks, active)):
            masked = F.log_softmax(values.masked_fill(~mask, -1e4), -1)
            total += masked.gather(-1, targets[..., field].unsqueeze(-1)).squeeze(-1) * weight
        return total.sum(1) / valid.sum(1).clamp_min(1)


@torch.no_grad()
def sample_graphs(model, record, formulas, device, trajectories=32, seed=42):
    rng = torch.Generator(device=device).manual_seed(seed)
    candidates, counters = {}, Counter()
    for formula in formulas:
        states = [CompleteState(formula) for _ in range(trajectories)]
        traces = [[(START, 0, 0, 0)] for _ in states]
        for state in states:
            state.apply((START, 0, 0, 0))
        histories = [[state.snapshot()] for state in states]
        context, memory, padding = model.condition([record] * trajectories, device)
        live = list(range(trajectories))
        for step in range(MAX_STEPS - 1):
            if not live:
                break
            # Retain exact replay snapshots; decoder prefixes remain unchanged.
            tokens = torch.tensor([traces[i] for i in live], device=device)
            snapshots = [histories[i] for i in live]
            packed = tuple(torch.as_tensor(np.stack([[s[k] for s in row] for row in snapshots]), device=device) for k in range(4))
            counts = torch.tensor([formula] * len(live), device=device, dtype=torch.float32)
            h = model.hidden(tokens, packed, counts, context[live], memory[live], padding[live])[:, -1:]
            current = tuple(x[:, -1:] for x in packed)
            samples = np.zeros((len(live), 4), dtype=np.int64)
            dead = np.zeros(len(live), dtype=bool)
            # Batch each conditional action field to avoid a GPU synchronization
            # and a complete pointer-head computation for every trajectory.
            for field, size in enumerate((5, 18, 4, MAX_ATOMS)):
                masks = np.zeros((len(live), size), dtype=bool)
                for local, global_index in enumerate(live):
                    state = states[global_index]
                    kind, atom, bond = (int(v) for v in samples[local, :3])
                    if field == 0:
                        kind = ADD
                    inactive = field and (kind == STOP or (kind == CLOSE and field == 1)
                                          or (len(state.types) == 0 and field >= 2))
                    bits = 1 if inactive or dead[local] else state.masks((kind, atom, bond, 0))[0][field]
                    if not bits:
                        dead[local] = True
                        counters["dead_end"] += 1
                        bits = 1
                    masks[local] = [bool(bits & (1 << j)) for j in range(size)]
                values = model.fields(h, torch.as_tensor(samples[:, 1:2], device=device),
                                      torch.as_tensor(samples[:, 2:3], device=device), current)[field][:, 0]
                mask = torch.as_tensor(masks, device=device)
                probability = F.softmax(values.masked_fill(~mask, -1e4), -1)
                samples[:, field] = torch.multinomial(probability, 1, generator=rng).flatten().cpu().numpy()
            next_live = []
            for local, global_index in enumerate(live):
                state = states[global_index]
                if dead[local]:
                    continue
                token = tuple(int(v) for v in samples[local])
                state.apply(token)
                traces[global_index].append(token)
                if token[0] == STOP:
                    graph = state.graph()
                    if graph is not None:
                        candidates[graph.identity] = graph
                        counters["valid_finished"] += 1
                    else:
                        counters["sanitization_failed"] += 1
                else:
                    histories[global_index].append(state.snapshot())
                    next_live.append(global_index)
            live = next_live
        counters["step_limit"] += len(live)
    return list(candidates.values()), dict(counters)
