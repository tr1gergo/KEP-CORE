from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import matplotlib
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pulp

VertexId = int
PlayerId = int
CycleId = int

AdjOut = Dict[VertexId, List[VertexId]]
AdjIn = Dict[VertexId, List[VertexId]]


@dataclass(frozen=True)
class Cycle:
    id: CycleId
    vertices: Tuple[VertexId, ...]
    length: int
    players_in_cycle: Set[PlayerId]
    player_counts: Dict[PlayerId, int]
    has_altruist: bool
    altruist_count: int
    non_altruist_count: int
    same_blood_edges: int
    hard_match_score: float


@dataclass
class CycleDB:
    cycles: List[Cycle]
    by_vertex: Dict[VertexId, List[CycleId]]
    by_player: Dict[PlayerId, List[CycleId]]
    has_altruist: bool
    altruist_vertices: Tuple[VertexId, ...] = ()

    @property
    def altruist_vertex(self) -> Optional[VertexId]:
        return self.altruist_vertices[0] if self.altruist_vertices else None


@dataclass
class Partition:
    owner_of: Dict[VertexId, PlayerId]
    vertices_of_player: Dict[PlayerId, List[VertexId]]
    players: List[PlayerId]


@dataclass(frozen=True)
class GraphFeatures:
    donor_bloodtype: Dict[VertexId, str]
    patient_bloodtype: Dict[VertexId, str]


def build_graph_features(instance) -> GraphFeatures:
    """Extract blood-type mappings for donors and their paired patients."""
    donors = getattr(instance, "donors", [])
    recipients = getattr(instance, "recipients", [])
    recipient_bloodtype: Dict[int, str] = {
        int(r.recipient_id): r.bloodtype or "" for r in recipients
    }
    donor_bloodtype: Dict[VertexId, str] = {
        int(d.donor_id): d.bloodtype or "" for d in donors
    }
    patient_bloodtype: Dict[VertexId, str] = {}
    for donor in donors:
        if getattr(donor, "source_patient_ids", ()):
            patient_id = int(donor.source_patient_ids[0])
            patient_bloodtype[int(donor.donor_id)] = recipient_bloodtype.get(
                patient_id, ""
            )
    return GraphFeatures(
        donor_bloodtype=donor_bloodtype,
        patient_bloodtype=patient_bloodtype,
    )


def compute_edge_same_blood(
    adj_out: Mapping[VertexId, Sequence[VertexId]],
    features: GraphFeatures,
) -> Dict[Tuple[VertexId, VertexId], int]:
    """Mark edges whose donor-recipient blood types match."""
    edge_flags: Dict[Tuple[VertexId, VertexId], int] = {}
    for u, neighbors in adj_out.items():
        donor_bt = features.donor_bloodtype.get(u, "")
        for v in neighbors:
            patient_bt = features.patient_bloodtype.get(v, "")
            edge_flags[(u, v)] = int(bool(donor_bt) and donor_bt == patient_bt)
    return edge_flags


def build_compat_graph(
    instance,
    num_patients: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[VertexId], AdjOut, AdjIn, Dict[VertexId, List[VertexId]]]:
    """
    Build the compatibility graph on non-altruistic vertices.

    Parameters
    ----------
    instance : object with `donors` attribute (Iterable[DonorRecord])
        Only donors with a non-empty `source_patient_ids` list are treated as vertices.
    num_patients : int, optional
        Sample this many vertices uniformly without replacement, yielding the induced subgraph.
    rng : numpy.random.Generator, optional
        Controls deterministic sampling.

    Returns
    -------
    vertices : list[int]
        Selected non-altruist vertex ids.
    adj_out : dict[int, list[int]]
        Outgoing adjacency restricted to selected vertices.
    adj_in : dict[int, list[int]]
        Incoming adjacency (for convenience in two-cycle detection).
    altruist_adj : dict[int, list[int]]
        Mapping of altruist donor ids to the compatible vertices retained in the sample.
    """
    donors = getattr(instance, "donors", [])
    non_altruists = [d for d in donors if getattr(d, "source_patient_ids", ())]
    if num_patients is not None and num_patients > len(non_altruists):
        raise ValueError("num_patients exceeds available non-altruistic pairs")

    vertex_ids = [int(d.donor_id) for d in non_altruists]
    patient_id_to_vertex: Dict[int, VertexId] = {}
    for donor in non_altruists:
        sources = getattr(donor, "source_patient_ids", ())
        if not sources:
            continue
        patient_id_to_vertex[int(sources[0])] = int(donor.donor_id)

    if num_patients is not None:
        rng = rng or np.random.default_rng()
        chosen = rng.choice(vertex_ids, size=num_patients, replace=False)
        selected_vertices = sorted(int(v) for v in chosen)
    else:
        selected_vertices = sorted(vertex_ids)

    vertex_set = set(selected_vertices)
    adj_out: AdjOut = {v: [] for v in selected_vertices}

    for donor in non_altruists:
        donor_id = int(donor.donor_id)
        if donor_id not in vertex_set:
            continue
        matches = getattr(donor, "matches", ())
        seen = set()
        for match in matches:
            if isinstance(match, tuple):
                recipient_id = int(match[0])
            else:
                recipient_id = int(getattr(match, "recipient", match.find("recipient").text))
            target_vertex = patient_id_to_vertex.get(recipient_id)
            if target_vertex is None or target_vertex not in vertex_set or target_vertex == donor_id:
                continue
            if target_vertex not in seen:
                adj_out[donor_id].append(target_vertex)
                seen.add(target_vertex)

    adj_in: AdjIn = {v: [] for v in selected_vertices}
    for u, neighbors in adj_out.items():
        for v in neighbors:
            if v in adj_in:
                adj_in[v].append(u)

    altruist_adj = compute_altruist_edges(instance, selected_vertices)

    return selected_vertices, adj_out, adj_in, altruist_adj


def compute_altruist_edges(
    instance,
    selected_vertices: Sequence[VertexId],
) -> Dict[VertexId, List[VertexId]]:
    """
    Extract altruist → vertex compatibility restricted to the selected vertices.

    Parameters
    ----------
    instance : object with `donors`
        Parsed instance providing donor records.
    selected_vertices : Sequence[int]
        Non-altruist vertex ids retained in the working graph.

    Returns
    -------
    dict[int, list[int]]
        Mapping altruist donor id -> sorted list of reachable vertex ids.
    """
    donors = getattr(instance, "donors", [])
    vertex_set = set(int(v) for v in selected_vertices)
    patient_to_vertex: Dict[int, VertexId] = {}
    for donor in donors:
        sources = getattr(donor, "source_patient_ids", ())
        if sources:
            patient_to_vertex[int(sources[0])] = int(donor.donor_id)

    altruist_edges: Dict[VertexId, List[VertexId]] = {}
    for donor in donors:
        if getattr(donor, "source_patient_ids", ()):
            continue
        donor_id = int(donor.donor_id)
        matches = getattr(donor, "matches", ())
        targets: Set[VertexId] = set()
        for match in matches:
            if isinstance(match, tuple):
                recipient_id = int(match[0])
            else:
                recipient_id = int(getattr(match, "recipient", match.find("recipient").text))
            vertex = patient_to_vertex.get(recipient_id)
            if vertex is not None and vertex in vertex_set:
                targets.add(vertex)
        altruist_edges[donor_id] = sorted(targets)
    return altruist_edges


def make_partition(
    vertices: List[VertexId],
    num_players: int,
    var_size: int,
    rng: Optional[np.random.Generator] = None,
) -> Partition:
    """
    Assign each vertex to a unique player.

    Parameters
    ----------
    vertices : list[int]
        Non-altruistic vertex identifiers.
    num_players : int
        Number of hospitals/players.
    var_size : int
        Controls imbalance: 0 (near equal), 1 (moderate), 2 (heavy tail).
    rng : numpy.random.Generator, optional
        Controls deterministic shuffling.
    """
    if num_players <= 0:
        raise ValueError("num_players must be positive")
    if var_size not in (0, 1, 2):
        raise ValueError("var_size must be 0, 1, or 2")

    rng = rng or np.random.default_rng()
    shuffled = list(vertices)
    rng.shuffle(shuffled)
    n = len(shuffled)

    if num_players == 1:
        sizes = [n]
    elif var_size == 0:
        base = n // num_players
        remainder = n % num_players
        sizes = [base + (1 if i < remainder else 0) for i in range(num_players)]
    else:
        alpha = 5.0 if var_size == 1 else 0.5
        proportions = rng.dirichlet([alpha] * num_players)
        raw = proportions * n
        floors = np.floor(raw).astype(int)
        remainder = n - int(floors.sum())
        if remainder > 0:
            fractional = raw - floors
            order = np.argsort(-fractional)
            for idx in order[:remainder]:
                floors[idx] += 1
        elif remainder < 0:
            order = np.argsort(floors)
            for idx in order[: (-remainder)]:
                if floors[idx] > 0:
                    floors[idx] -= 1
        sizes = floors.tolist()

    owner_of: Dict[VertexId, PlayerId] = {}
    vertices_of_player: Dict[PlayerId, List[VertexId]] = {p: [] for p in range(num_players)}

    offset = 0
    for player_id, size in enumerate(sizes):
        chunk = shuffled[offset : offset + size]
        offset += size
        vertices_of_player[player_id] = chunk
        for v in chunk:
            owner_of[v] = player_id

    return Partition(owner_of=owner_of, vertices_of_player=vertices_of_player, players=list(range(num_players)))

# Utility: randomly delete inter-player edges with probability p (in [0,1])
import numpy as np

def delete_cross_player_edges(adj_out, partition, p: float = 0.0, rng: np.random.Generator | None = None):
    """
    Return a new adjacency dict where edges between different players are independently
    deleted with probability p. Within-player edges remain intact.

    Parameters
    - adj_out: dict[int, list[int]] outgoing adjacency lists
    - partition: Partition object with `owner_of` mapping vertex -> player
    - p: deletion probability for cross-player edges (0 <= p <= 1)
    - rng: numpy Generator for reproducibility (optional)

    Returns
    - new_adj_out: dict[int, list[int]] with filtered edges
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    if p == 0.0:
        # No deletion; return a shallow copy of lists to avoid accidental mutation downstream
        return {u: list(vs) for u, vs in adj_out.items()}

    generator = rng if rng is not None else np.random.default_rng()
    owner_of = partition.owner_of

    new_adj_out: dict[int, list[int]] = {}
    for u, neighbors in adj_out.items():
        u_owner = owner_of[u]
        kept_neighbors: list[int] = []
        for v in neighbors:
            if owner_of[v] == u_owner:
                kept_neighbors.append(v)
            else:
                # Keep with probability (1 - p)
                if generator.random() >= p:
                    kept_neighbors.append(v)
        new_adj_out[u] = kept_neighbors

    return new_adj_out


def enumerate_cycles(
    vertices: Sequence[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    altruist_vertex: Optional[VertexId] = None,
    edge_same_blood: Optional[Mapping[Tuple[VertexId, VertexId], int]] = None,
    vertex_hardness: Optional[Mapping[VertexId, float]] = None,
) -> CycleDB:
    """
    Enumerate all directed cycles of length ≤ Delta.

    Parameters
    ----------
    vertices : Sequence[int]
        Vertices currently in the graph (non-altruists plus any altruists already added).
    adj_out : dict[int, list[int]]
        Outgoing adjacency lists.
    partition : Partition
        Vertex-to-player ownership.
    Delta : int
        Maximum allowed cycle length (2 or 3).
    altruist_vertex : int or iterable, optional
        Single altruist id or iterable of ids already present.
    """
    if Delta not in (2, 3):
        raise ValueError("Delta must be 2 or 3")

    if altruist_vertex is None:
        altruists: Tuple[VertexId, ...] = ()
    elif isinstance(altruist_vertex, (list, tuple, set)):
        altruists = tuple(int(a) for a in altruist_vertex)
    else:
        altruists = (int(altruist_vertex),)

    vertex_set = set(int(v) for v in vertices)
    all_vertices = sorted(vertex_set.union(altruists))
    adj_sets = {u: set(adj_out.get(u, [])) for u in all_vertices}
    edge_same_blood = edge_same_blood or {}
    vertex_hardness = vertex_hardness or {}

    cycles: List[Cycle] = []
    by_vertex: Dict[VertexId, List[CycleId]] = defaultdict(list)
    by_player: Dict[PlayerId, List[CycleId]] = defaultdict(list)
    owner = partition.owner_of

    def add_cycle(order: Tuple[VertexId, ...]) -> None:
        cid = len(cycles)
        altruist_count = sum(1 for v in order if v in altruists)
        edges = list(zip(order, order[1:] + order[:1]))
        same_blood_edges = sum(edge_same_blood.get(edge, 0) for edge in edges)
        hardness_score = float(max(vertex_hardness.get(v, 0.0) for v in order))
        player_counts: Dict[PlayerId, int] = defaultdict(int)
        for v in order:
            if v in owner:
                player_counts[owner[v]] += 1
        players_in_cycle = set(player_counts.keys())
        cycle = Cycle(
            id=cid,
            vertices=order,
            length=len(order),
            players_in_cycle=players_in_cycle,
            player_counts=dict(player_counts),
            has_altruist=altruist_count > 0,
            altruist_count=altruist_count,
            non_altruist_count=len(order) - altruist_count,
            same_blood_edges=same_blood_edges,
            hard_match_score=hardness_score,
        )
        cycles.append(cycle)
        for v in order:
            by_vertex[v].append(cid)
        for p in players_in_cycle:
            by_player[p].append(cid)

    # 2-cycles
    for u in all_vertices:
        for v in adj_out.get(u, []):
            if v not in adj_sets:
                continue
            if u >= v:
                continue
            if u in adj_sets.get(v, set()):
                add_cycle((u, v))

    if Delta == 3:
        for u in all_vertices:
            for v in adj_out.get(u, []):
                if v == u or v not in adj_sets:
                    continue
                for w in adj_out.get(v, []):
                    if w in (u, v) or w not in adj_sets:
                        continue
                    if u not in adj_sets.get(w, set()):
                        continue
                    if min(u, v, w) != u:
                        continue
                    if v >= w:
                        continue
                    add_cycle((u, v, w))

    has_altruist = bool(altruists)
    return CycleDB(
        cycles=cycles,
        by_vertex=dict(by_vertex),
        by_player=dict(by_player),
        has_altruist=has_altruist,
        altruist_vertices=altruists,
    )


def compute_vertex_hardness(adj_out: Mapping[VertexId, Sequence[VertexId]]) -> Dict[VertexId, float]:
    """Assign hardness scores inversely proportional to total incident edges."""
    nodes: Set[VertexId] = set(adj_out.keys())
    for neighbors in adj_out.values():
        nodes.update(neighbors)
    in_deg: Dict[VertexId, int] = {v: 0 for v in nodes}
    for neighbors in adj_out.values():
        for v in neighbors:
            in_deg[v] = in_deg.get(v, 0) + 1
    hardness = {}
    for v in nodes:
        degree = in_deg.get(v, 0)
        hardness[v] = 0.0 if degree == 0 else 1.0 / degree
    return hardness


def compute_player_utilities(solution: Set[CycleId], cycle_db: CycleDB) -> Dict[PlayerId, int]:
    """Return u_i = Σ_{c∈solution} α_{c,i}."""
    utilities: Dict[PlayerId, int] = defaultdict(int)
    for cid in solution:
        cycle = cycle_db.cycles[cid]
        for player, count in cycle.player_counts.items():
            utilities[player] += count
    return dict(utilities)


def cycles_allowed_by_coalition(cycle_db: CycleDB, coalition: Set[PlayerId]) -> List[CycleId]:
    """Cycles fully controlled by coalition (no altruists)."""
    allowed: List[CycleId] = []
    for cycle in cycle_db.cycles:
        if cycle.has_altruist:
            continue
        if all(player in coalition for player in cycle.players_in_cycle):
            allowed.append(cycle.id)
    return allowed


def make_pulp_solver(
    solver: str,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    msg: bool = False,
) -> pulp.LpSolver:
    """Return a configured PuLP solver."""
    solver_name = (solver or "CBC").upper()
    if solver_name == "GUROBI":
        try:
            return pulp.GUROBI(
                msg=msg,
                timeLimit=time_limit,
            )
        except pulp.PulpSolverError:
            pass
    gap_kwargs = {}
    if mip_gap is not None:
        gap_kwargs["fracGap"] = mip_gap
    return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, **gap_kwargs)


def _cycle_weight(cycle: Cycle, has_altruist_mode: bool) -> int:
    return cycle.non_altruist_count if has_altruist_mode else cycle.length


def _solve_cycle_ip(
    cycle_db: CycleDB,
    Delta: int,
    partition: Partition,
    solver: str,
    cuts: Optional[List[Dict[str, object]]] = None,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    min_real_transplants: Optional[int] = None,
) -> Tuple[Set[CycleId], int]:
    cycles = [c for c in cycle_db.cycles if c.length <= Delta]
    problem = pulp.LpProblem("CycleSelection", pulp.LpMaximize)
    y_vars = {
        cycle.id: pulp.LpVariable(f"y_{cycle.id}", lowBound=0, upBound=1, cat="Binary")
        for cycle in cycles
    }
    problem += pulp.lpSum(
        y_vars[cycle.id] * _cycle_weight(cycle, cycle_db.has_altruist) for cycle in cycles
    )
    for vertex in cycle_db.by_vertex.keys():
        relevant = [y_vars[cid] for cid in cycle_db.by_vertex.get(vertex, []) if cid in y_vars]
        if relevant:
            problem += pulp.lpSum(relevant) <= 1, f"disjoint_v{vertex}"

    if min_real_transplants is not None and cycles:
        problem += pulp.lpSum(y_vars[cycle.id] * cycle.non_altruist_count for cycle in cycles) >= int(
            min_real_transplants
        ), "min_real_transplants"

    if cuts:
        for idx, cut in enumerate(cuts):
            coalition = set(cut["coalition"])
            rhs = int(cut["rhs"])
            lhs = []
            for cycle in cycles:
                weight = sum(cycle.player_counts.get(player, 0) for player in coalition)
                if weight:
                    lhs.append(y_vars[cycle.id] * weight)
            if lhs:
                problem += pulp.lpSum(lhs) >= rhs, f"cut_{idx}"

    solver_instance = make_pulp_solver(solver, time_limit=time_limit, mip_gap=mip_gap)
    problem.solve(solver_instance)
    status = problem.status
    selected = {
        cid for cid, var in y_vars.items() if var.value() is not None and var.value() > 0.5
    }
    return selected, status


def solve_lexicographic_cycle_cover(
    cycle_db: CycleDB,
    Delta: int,
    partition: Partition,
    solver: str = "GUROBI",
    warm_start: Optional[Set[CycleId]] = None,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
) -> Tuple[Set[CycleId], Dict[str, float], int]:
    """
    Solve the disjoint-cycle IP with four-stage lexicographic optimization.
    """
    cycles = [c for c in cycle_db.cycles if c.length <= Delta]
    objectives = {
        "transplants": {c.id: c.non_altruist_count for c in cycles},
        "cycle_count": {c.id: 1 for c in cycles},
        "same_blood": {c.id: c.same_blood_edges for c in cycles},
        "hard_match": {c.id: c.hard_match_score for c in cycles},
    }
    if not cycles:
        return set(), {k: 0.0 for k in objectives}, pulp.LpStatusOptimal

    problem = pulp.LpProblem("Lexicographic_Cycle_Cover", pulp.LpMaximize)
    y_vars = {
        cycle.id: pulp.LpVariable(f"y_{cycle.id}", lowBound=0, upBound=1, cat="Binary")
        for cycle in cycles
    }
    for vertex in partition.owner_of.keys():
        relevant = [y_vars[cid] for cid in cycle_db.by_vertex.get(vertex, []) if cid in y_vars]
        if relevant:
            problem += pulp.lpSum(relevant) <= 1, f"lex_disjoint_{vertex}"

    warm = warm_start or set()
    objective_values: Dict[str, float] = {}
    stage_order = ["transplants", "cycle_count", "same_blood", "hard_match"]

    for stage_index, stage_name in enumerate(stage_order):
        for cid, var in y_vars.items():
            var.setInitialValue(1 if cid in warm else 0)
        weights = objectives[stage_name]
        expr = pulp.lpSum(y_vars[cid] * weights.get(cid, 0.0) for cid in y_vars)
        problem.setObjective(expr)
        solver_instance = make_pulp_solver(
            solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        problem.solve(solver_instance)
        status = problem.status
        objective_values[stage_name] = pulp.value(expr) if problem.status in (
            pulp.LpStatusOptimal,
            pulp.LpStatusNotSolved,
        ) else 0.0
        if status not in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved):
            warm = {cid for cid, var in y_vars.items() if var.value() and var.value() > 0.5}
            return warm, objective_values, status
        warm = {cid for cid, var in y_vars.items() if var.value() and var.value() > 0.5}
        if stage_index < len(stage_order) - 1:
            problem += expr == objective_values[stage_name], f"lex_fix_{stage_index}_{stage_name}"

    return warm, objective_values, pulp.LpStatusOptimal


def lexicographic_core_search(
    vertices: List[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    graph_features: GraphFeatures,
    max_coal_size: int = 3,
    solver: str = "GUROBI",
    altruist_edges: Optional[Mapping[VertexId, Sequence[VertexId]]] = None,
    max_added_altruists: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """
    Iteratively apply lexicographic optimization, adding altruists only when needed for core feasibility.
    """
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}
    working_adj = {u: list(neigh) for u, neigh in adj_out.items()}
    edge_same_blood = compute_edge_same_blood(working_adj, graph_features)
    current_vertices = list(vertices)
    altruists_added: List[VertexId] = []
    available_altruists = [
        altruist
        for altruist in sorted(altruist_edges.keys())
        if altruist not in working_adj
    ]
    warm_start: Optional[Set[CycleId]] = None
    history: List[Dict[str, object]] = []

    while True:
        vertex_hardness = compute_vertex_hardness(working_adj)
        cycle_db = enumerate_cycles(
            current_vertices,
            working_adj,
            partition,
            Delta,
            altruists_added,
            edge_same_blood=edge_same_blood,
            vertex_hardness=vertex_hardness,
        )
        selection, objectives, status = solve_lexicographic_cycle_cover(
            cycle_db,
            Delta,
            partition,
            solver=solver,
            warm_start=warm_start,
        )
        history.append(
            {
                "altruists": len(altruists_added),
                "status": status,
                "objectives": objectives,
                "solution_size": len(selection),
            }
        )

        in_core, _, _ = core_verification(
            selection, cycle_db, partition, max_coal_size, Delta, solver=solver
        )
        if in_core:
            return {
                "solution": selection,
                "altruists_added": len(altruists_added),
                "final_in_core": True,
                "objective_tiers": objectives,
                "player_utilities": compute_player_utilities(selection, cycle_db),
                "history": history,
            }

        if not available_altruists or (
            max_added_altruists is not None and len(altruists_added) >= max_added_altruists
        ):
            return {
                "solution": selection,
                "altruists_added": len(altruists_added),
                "final_in_core": False,
                "objective_tiers": objectives,
                "player_utilities": compute_player_utilities(selection, cycle_db),
                "history": history,
            }

        new_altruist = available_altruists.pop(0)
        targets = [
            target
            for target in altruist_edges.get(new_altruist, [])
            if target in current_vertices
        ]
        _add_altruist_vertex(
            working_adj,
            current_vertices,
            new_altruist,
            targets,
            edge_same_blood=edge_same_blood,
            graph_features=graph_features,
            rng=rng,
        )
        altruists_added.append(new_altruist)
        current_vertices.append(new_altruist)
        warm_start = selection


def max_size_solution(
    cycle_db: CycleDB,
    Delta: int,
    partition: Partition,
    solver: str = "GUROBI",
) -> Set[CycleId]:
    """Solve the base IP maximising total transplants without altruists."""
    solution, status = _solve_cycle_ip(cycle_db, Delta, partition, solver)
    if status not in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved):
        raise RuntimeError("Failed to optimise maximum size solution")
    return solution


def core_verification(
    solution: Set[CycleId],
    cycle_db: CycleDB,
    partition: Partition,
    max_coal_size: int,
    Delta: int,
    solver: str = "GUROBI",
) -> Tuple[bool, Optional[Set[PlayerId]], Optional[Dict[PlayerId, int]]]:
    """
    Check whether the given solution is in the core w.r.t. coalitions of size ≤ max_coal_size.
    """
    if max_coal_size <= 0:
        raise ValueError("max_coal_size must be positive")
    utilities = {player: 0 for player in partition.players}
    utilities.update(compute_player_utilities(solution, cycle_db))

    players = partition.players
    for size in range(1, min(max_coal_size, len(players)) + 1):
        for coalition_tuple in combinations(players, size):
            coalition = set(coalition_tuple)
            candidate_cycles = cycles_allowed_by_coalition(cycle_db, coalition)
            if not candidate_cycles:
                continue
            sub_cycles = [cycle_db.cycles[cid] for cid in candidate_cycles if cycle_db.cycles[cid].length <= Delta]
            if not sub_cycles:
                continue
            problem = pulp.LpProblem("CoalitionCheck", pulp.LpMaximize)
            y_vars = {
                cycle.id: pulp.LpVariable(f"yc_{cycle.id}", lowBound=0, upBound=1, cat="Binary")
                for cycle in sub_cycles
            }
            problem += pulp.lpSum(y_vars[cycle.id] * cycle.length for cycle in sub_cycles)
            coalition_vertices = {
                v
                for player in coalition
                for v in partition.vertices_of_player.get(player, [])
            }
            for vertex in coalition_vertices:
                relevant = [
                    y_vars[cid]
                    for cid in cycle_db.by_vertex.get(vertex, [])
                    if cid in y_vars
                ]
                if relevant:
                    problem += pulp.lpSum(relevant) <= 1, f"coal_disjoint_{vertex}"
            for player in coalition:
                rhs = utilities.get(player, 0) + 1
                lhs = [
                    y_vars[cycle.id] * cycle.player_counts.get(player, 0)
                    for cycle in sub_cycles
                    if cycle.player_counts.get(player, 0)
                ]
                if lhs:
                    problem += pulp.lpSum(lhs) >= rhs, f"improve_{player}"
                else:
                    problem += pulp.lpSum([]) >= rhs, f"improve_{player}_zero"

            solver_instance = make_pulp_solver(solver)
            problem.solve(solver_instance)
            if problem.status == pulp.LpStatusOptimal:
                return False, coalition, utilities

    return True, None, utilities


def _add_altruist_vertex(
    adj_out: AdjOut,
    base_vertices: Sequence[VertexId],
    altruist_id: VertexId,
    targets: Sequence[VertexId],
    edge_same_blood: Optional[Dict[Tuple[VertexId, VertexId], int]] = None,
    graph_features: Optional[GraphFeatures] = None,
    rng: Optional[np.random.Generator] = None,
) -> VertexId:
    if altruist_id in adj_out:
        raise ValueError(f"Altruist vertex {altruist_id} already present")
    adj_out[altruist_id] = []
    # allow closing cycles by letting existing vertices donate to altruist
    for v in base_vertices:
        neighbors = adj_out.setdefault(v, [])
        if altruist_id not in neighbors:
            neighbors.append(altruist_id)
            if edge_same_blood is not None:
                edge_same_blood[(v, altruist_id)] = 0
    if targets:
        for target in targets:
            if target in base_vertices and target not in adj_out[altruist_id]:
                adj_out[altruist_id].append(target)
                if edge_same_blood is not None:
                    donor_bt = graph_features.donor_bloodtype.get(altruist_id, "") if graph_features else ""
                    patient_bt = graph_features.patient_bloodtype.get(target, "") if graph_features else ""
                    edge_same_blood[(altruist_id, target)] = int(bool(donor_bt) and donor_bt == patient_bt)
    elif rng is not None and base_vertices:
        target_count = min(len(base_vertices), max(1, len(base_vertices) // 5))
        sampled = rng.choice(base_vertices, size=target_count, replace=False)
        for target in sampled:
            adj_out[altruist_id].append(int(target))
            if edge_same_blood is not None:
                donor_bt = graph_features.donor_bloodtype.get(altruist_id, "") if graph_features else ""
                patient_bt = graph_features.patient_bloodtype.get(int(target), "") if graph_features else ""
                edge_same_blood[(altruist_id, int(target))] = int(bool(donor_bt) and donor_bt == patient_bt)
    return altruist_id


def strong_core_verification(
    solution: Set[CycleId],
    cycle_db: CycleDB,
    partition: Partition,
    max_coal_size: int,
    Delta: int,
    solver: str = "GUROBI",
) -> Tuple[bool, Optional[Set[PlayerId]], Optional[Dict[PlayerId, int]]]:
    """
    Strong-core check: For every coalition S (|S| ≤ max_coal_size), test if there exists
    a deviation on the subgraph induced by S where:
      - each player i∈S gets utility ≥ their baseline utility in the input solution;
      - the total utility of S is ≥ baseline total + 1.

    Returns (in_strong_core, blocking_coalition, baseline_utilities_if_blocked).
    """
    if max_coal_size <= 0:
        raise ValueError("max_coal_size must be positive")

    baseline_utilities = {player: 0 for player in partition.players}
    baseline_utilities.update(compute_player_utilities(solution, cycle_db))

    players = partition.players
    for size in range(1, min(max_coal_size, len(players)) + 1):
        for coalition_tuple in combinations(players, size):
            coalition = set(coalition_tuple)
            candidate_cycles = cycles_allowed_by_coalition(cycle_db, coalition)
            if not candidate_cycles:
                continue
            sub_cycles = [cycle_db.cycles[cid] for cid in candidate_cycles if cycle_db.cycles[cid].length <= Delta]
            if not sub_cycles:
                continue

            problem = pulp.LpProblem("StrongCoalitionCheck", pulp.LpMaximize)
            y_vars = {
                cycle.id: pulp.LpVariable(f"ys_{cycle.id}", lowBound=0, upBound=1, cat="Binary")
                for cycle in sub_cycles
            }
            # Any objective is fine; maximize covered vertices
            problem += pulp.lpSum(y_vars[cycle.id] * cycle.length for cycle in sub_cycles)

            # At most one cycle per vertex in S
            coalition_vertices = {
                v
                for player in coalition
                for v in partition.vertices_of_player.get(player, [])
            }
            for vertex in coalition_vertices:
                relevant = [y_vars[cid] for cid in cycle_db.by_vertex.get(vertex, []) if cid in y_vars]
                if relevant:
                    problem += pulp.lpSum(relevant) <= 1, f"strong_disjoint_{vertex}"

            # Per-player constraints: utility_i >= baseline_i (no +1)
            for player in coalition:
                rhs_i = baseline_utilities.get(player, 0)
                lhs_i = [y_vars[cycle.id] * cycle.player_counts.get(player, 0) for cycle in sub_cycles if cycle.player_counts.get(player, 0)]
                if lhs_i:
                    problem += pulp.lpSum(lhs_i) >= rhs_i, f"strong_player_{player}"
                else:
                    # No cycles contributing to this player ⇒ must have baseline 0 to be feasible
                    problem += pulp.lpSum([]) >= rhs_i, f"strong_player_{player}_zero"

            # Coalition total ≥ baseline total + 1
            baseline_sum = sum(baseline_utilities.get(player, 0) for player in coalition)
            lhs_sum_terms = []
            for cycle in sub_cycles:
                weight = sum(cycle.player_counts.get(player, 0) for player in coalition)
                if weight:
                    lhs_sum_terms.append(y_vars[cycle.id] * weight)
            if lhs_sum_terms:
                problem += pulp.lpSum(lhs_sum_terms) >= (baseline_sum + 1), "strong_total_improve"
            else:
                problem += pulp.lpSum([]) >= (baseline_sum + 1), "strong_total_improve_zero"

            solver_instance = make_pulp_solver(solver)
            problem.solve(solver_instance)
            if problem.status == pulp.LpStatusOptimal:
                return False, coalition, baseline_utilities

    return True, None, baseline_utilities


def strong_core_heuristic(
    vertices: List[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    solver: str = "GUROBI",
    max_coal_size: int = 3,
    max_altruists: int = 10,
    rng: Optional[np.random.Generator] = None,
    altruist_edges: Optional[Mapping[VertexId, Sequence[VertexId]]] = None,
) -> Dict[str, object]:
    """
    Heuristic identical to core_heuristic but using strong_core_verification.
    Adds altruists only if the IP with accumulated cuts becomes infeasible.
    """
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}

    existing_ids = set(vertices) | set(adj_out.keys())
    available_altruists = sorted(a for a in altruist_edges if a not in existing_ids)

    working_adj = {u: list(neigh) for u, neigh in adj_out.items()}
    current_vertices = list(vertices)
    altruists: List[VertexId] = []
    cuts: List[Dict[str, object]] = []

    # Initial cycle DB and solve
    cycle_db = enumerate_cycles(current_vertices + altruists, working_adj, partition, Delta, altruists)
    solution, status = _solve_cycle_ip(cycle_db, Delta, partition, solver)

    # Accept only Optimal OR NotSolved with a usable incumbent
    if not (status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and solution)):
        return {
            "solution": set(),
            "altruists_added": 0,
            "cuts_used": [],
            "final_in_core": False,
            "player_utilities": {},
            "objective_value": 0,
        }

    min_real_transplants = sum(
        cycle_db.cycles[cid].non_altruist_count for cid in solution
    )

    best_solution = solution
    final_in_core = False
    terminated = False

    while True:
        in_core, blocking_coalition, utilities = strong_core_verification(
            solution, cycle_db, partition, max_coal_size, Delta, solver=solver
        )
        if in_core:
            best_solution = solution
            final_in_core = True
            break

        # Add coalition cut: total utility of S must be at least previous + 1
        assert blocking_coalition is not None
        rhs = sum(utilities.get(player, 0) for player in blocking_coalition) + 1
        cuts.append({"coalition": sorted(blocking_coalition), "rhs": rhs})

        # Re-solve with cuts; only add altruists if the model becomes infeasible
        while True:
            solution, status = _solve_cycle_ip(
                cycle_db,
                Delta,
                partition,
                solver,
                cuts=cuts,
                min_real_transplants=min_real_transplants,
            )

            # (1) Accept solution when available
            if status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and solution):
                best_solution = solution
                break

            # (2) Infeasible => add altruist (if available) and retry
            if status == pulp.LpStatusInfeasible:
                if len(altruists) >= max_altruists:
                    terminated = True
                    solution = best_solution
                    break

                if available_altruists:
                    new_altruist = available_altruists.pop(0)
                    targets = altruist_edges.get(new_altruist, [])
                    _add_altruist_vertex(
                        working_adj,
                        current_vertices,
                        new_altruist,
                        [t for t in targets if t in current_vertices],
                    )
                else:
                    new_altruist = max(
                        set(working_adj.keys()).union(current_vertices).union(
                            v for nbrs in working_adj.values() for v in nbrs
                        ),
                        default=-1,
                    ) + 1
                    _add_altruist_vertex(working_adj, current_vertices, new_altruist, [], rng=rng)

                altruists.append(new_altruist)
                current_vertices.append(new_altruist)
                cycle_db = enumerate_cycles(current_vertices, working_adj, partition, Delta, altruists)
                continue

            terminated = True
            solution = best_solution
            break

        if terminated:
            break

    final_cycle_db = cycle_db
    player_utilities = compute_player_utilities(best_solution, final_cycle_db)
    objective_value = sum(
        _cycle_weight(final_cycle_db.cycles[cid], final_cycle_db.has_altruist)
        for cid in best_solution
    )
    return {
        "solution": best_solution,
        "altruists_added": len(altruists),
        "cuts_used": cuts,
        "final_in_core": final_in_core,
        "player_utilities": player_utilities,
        "objective_value": objective_value,
    }

def core_heuristic(
    vertices: List[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    solver: str = "GUROBI",
    max_coal_size: int = 3,
    max_altruists: int = 10,
    rng: Optional[np.random.Generator] = None,
    altruist_edges: Optional[Mapping[VertexId, Sequence[VertexId]]] = None,
) -> Dict[str, object]:
    """
    Heuristic that enforces coalition cuts and augments the graph with altruists if needed.

    Parameters
    ----------
    altruist_edges : Mapping[int, Sequence[int]], optional
        Precomputed altruist compatibility lists (donor_id -> list of target vertices)
        derived from the base instance. When provided, altruists are added in this
        order; otherwise synthetic altruists with random targets are generated.
        Altruists are only introduced when the IP with the accumulated cuts
        becomes infeasible, signalling that additional supply is required.
    """
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}

    # Altruists that can be added (real IDs from XML) and aren't already present
    existing_ids = set(vertices) | set(adj_out.keys())
    available_altruists = sorted(a for a in altruist_edges if a not in existing_ids)

    # Work on a mutable copy of the adjacency
    working_adj = {u: list(neigh) for u, neigh in adj_out.items()}
    current_vertices = list(vertices)
    altruists: List[VertexId] = []
    cuts: List[Dict[str, object]] = []

    # Initial cycle DB and solve
    cycle_db = enumerate_cycles(current_vertices + altruists, working_adj, partition, Delta, altruists)
    solution, status = _solve_cycle_ip(cycle_db, Delta, partition, solver)

    # Accept only Optimal OR NotSolved with a usable incumbent
    if not (status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and solution)):
        return {
            "solution": set(),
            "altruists_added": 0,
            "cuts_used": [],
            "final_in_core": False,
            "player_utilities": {},
            "objective_value": 0,
        }

    min_real_transplants = sum(
        cycle_db.cycles[cid].non_altruist_count for cid in solution
    )

    best_solution = solution
    final_in_core = False
    terminated = False

    while True:
        in_core, blocking_coalition, utilities = core_verification(
            solution, cycle_db, partition, max_coal_size, Delta, solver=solver
        )
        if in_core:
            best_solution = solution
            final_in_core = True
            break

        # Add coalition cut: total utility of S must be at least previous + 1
        assert blocking_coalition is not None
        rhs = sum(utilities.get(player, 0) for player in blocking_coalition) + 1
        cuts.append({"coalition": sorted(blocking_coalition), "rhs": rhs})

        # Re-solve with cuts; only add altruists if the model becomes infeasible
        while True:
            solution, status = _solve_cycle_ip(
                cycle_db,
                Delta,
                partition,
                solver,
                cuts=cuts,
                min_real_transplants=min_real_transplants,
            )

            # (1) Accept solution when available
            if status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and solution):
                best_solution = solution
                break  # leave inner loop, continue outer loop with updated solution

            # (2) Infeasible => add altruist (if available) and retry
            if status == pulp.LpStatusInfeasible:
                if len(altruists) >= max_altruists:
                    terminated = True
                    solution = best_solution
                    break

                if available_altruists:
                    new_altruist = available_altruists.pop(0)
                    targets = altruist_edges.get(new_altruist, [])
                    _add_altruist_vertex(
                        working_adj,
                        current_vertices,
                        new_altruist,
                        [t for t in targets if t in current_vertices],
                    )
                else:
                    # Fallback synthetic altruist id with no preset targets (optional)
                    new_altruist = max(
                        set(working_adj.keys()).union(current_vertices).union(
                            v for nbrs in working_adj.values() for v in nbrs
                        ),
                        default=-1,
                    ) + 1
                    _add_altruist_vertex(working_adj, current_vertices, new_altruist, [], rng=rng)

                # Book-keeping
                altruists.append(new_altruist)
                current_vertices.append(new_altruist)
                # Re-enumerate cycles including the new altruist and retry inner loop
                cycle_db = enumerate_cycles(current_vertices, working_adj, partition, Delta, altruists)
                continue  # keep trying with updated graph

            # (3) Any other status (e.g., Unbounded/Undefined, or NotSolved without incumbent)
            terminated = True
            solution = best_solution
            break  # break inner loop

        if terminated:
            break  # break outer loop

    final_cycle_db = cycle_db
    player_utilities = compute_player_utilities(best_solution, final_cycle_db)
    objective_value = sum(
        _cycle_weight(final_cycle_db.cycles[cid], final_cycle_db.has_altruist)
        for cid in best_solution
    )
    return {
        "solution": best_solution,
        "altruists_added": len(altruists),
        "cuts_used": cuts,
        "final_in_core": final_in_core,
        "player_utilities": player_utilities,
        "objective_value": objective_value,
    }



def visualize_compatibility_graph(
    vertices: Sequence[VertexId],
    adj_out: Mapping[VertexId, Sequence[VertexId]],
    partition: Partition,
    layout: str = "spring",
    seed: Optional[int] = None,
    ax=None,
    node_size: int = 500,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """
    Draw the compatibility graph with vertex colours indicating owning players.

    Parameters
    ----------
    vertices : Sequence[int]
        Vertex ids currently in the graph (non-altruists ± altruists).
    adj_out : Mapping[int, Sequence[int]]
        Outgoing adjacency list describing compatibility arcs.
    partition : Partition
        Vertex-to-player ownership information.
    layout : {'spring', 'kamada_kawai', 'circular'}
        Network layout algorithm.
    seed : int, optional
        Random seed passed to layout routine when supported.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure/axes pair is created if omitted.
    node_size : int
        Node marker size passed to NetworkX drawing util.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Handles for further customisation / saving.

    Notes
    -----
    Requires `networkx` and `matplotlib`. These are imported lazily so the
    rest of the module remains usable without plotting dependencies.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    for u in vertices:
        for v in adj_out.get(u, []):
            if v in vertices:
                G.add_edge(u, v)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        raise ValueError("Unsupported layout")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    players = sorted(partition.players)
    cmap = plt.cm.get_cmap("tab20", max(len(players), 1))
    player_colors = {player: cmap(idx) for idx, player in enumerate(players)}

    node_colors = []
    for v in vertices:
        owner = partition.owner_of.get(v)
        node_colors.append(player_colors.get(owner, (0.3, 0.3, 0.3)))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
        edgecolors="black",
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="->", arrowsize=12)

    handles = []
    labels = []
    for player in players:
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="", color=player_colors[player], markeredgecolor="black"
            )
        )
        labels.append(f"Player {player}")
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)

    ax.set_axis_off()
    ax.set_title("Compatibility Graph by Player")
    return fig, ax


# --- TU Core (Simple) with Altruists ----------------------------------------------------------

def _augment_with_altruists(
    vertices: Sequence[VertexId],
    adj_out: AdjOut,
    altruist_edges: Optional[Mapping[VertexId, Sequence[VertexId]]],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[VertexId], AdjOut, Set[VertexId]]:
    """
    Add altruist vertices to the graph once using provided adjacency. Partition remains unchanged.
    Returns updated (vertices, adj_out, altruist_vertex_ids).
    """
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}
    base_vertices = list(vertices)
    working_adj: AdjOut = {u: list(neigh) for u, neigh in adj_out.items()}

    existing_ids = set(base_vertices) | set(working_adj.keys())
    added_altruists: List[VertexId] = []
    for a in sorted(altruist_edges.keys()):
        if a in existing_ids:
            continue
        targets = [t for t in altruist_edges.get(a, []) if t in base_vertices]
        _add_altruist_vertex(working_adj, base_vertices, int(a), targets, rng=rng)
        base_vertices.append(int(a))
        added_altruists.append(int(a))

    return base_vertices, working_adj, set(added_altruists)


def _altruist_incidence_stats(
    cycle: Cycle, altruist_vertices: Set[VertexId]
) -> Tuple[int, int]:
    """
    Return (altruist_edge_penalty, distinct_altruists_in_cycle).
    - altruist_edge_penalty: number of edges of the cycle that are incident to an altruist vertex.
    - distinct_altruists_in_cycle: count of unique altruist vertices in the cycle.
    """
    verts = list(cycle.vertices)
    k = len(verts)
    distinct_altruists = {v for v in verts if v in altruist_vertices}
    penalty = 0
    for i in range(k):
        u = verts[i]
        v = verts[(i + 1) % k]
        if u in altruist_vertices or v in altruist_vertices:
            penalty += 1
    return penalty, len(distinct_altruists)


def _compute_opt_for_coalition(
    cycle_db: CycleDB,
    partition: Partition,
    coalition: Set[PlayerId],
    Delta: int,
    solver: str,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
) -> int:
    """Max vertices coalition S can cover without altruists, using cycles of length ≤ Delta."""
    candidate_cycles = cycles_allowed_by_coalition(cycle_db, coalition)
    if not candidate_cycles:
        return 0
    sub_cycles = [cycle_db.cycles[cid] for cid in candidate_cycles if cycle_db.cycles[cid].length <= Delta]
    if not sub_cycles:
        return 0

    problem = pulp.LpProblem("CoalitionOpt", pulp.LpMaximize)
    y_vars = {
        cyc.id: pulp.LpVariable(f"ys_{cyc.id}", lowBound=0, upBound=1, cat="Binary")
        for cyc in sub_cycles
    }
    # Objective: total vertices covered in S-only subgraph equals sum of cycle.length
    problem += pulp.lpSum(y_vars[cyc.id] * cyc.length for cyc in sub_cycles)

    coalition_vertices = {
        v for player in coalition for v in partition.vertices_of_player.get(player, [])
    }
    for vertex in coalition_vertices:
        relevant = [y_vars[cid] for cid in cycle_db.by_vertex.get(vertex, []) if cid in y_vars]
        if relevant:
            problem += pulp.lpSum(relevant) <= 1, f"coal_disjoint_{vertex}"

    solver_instance = make_pulp_solver(solver, time_limit=time_limit, mip_gap=mip_gap)
    problem.solve(solver_instance)
    if problem.status not in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved):
        return 0
    return int(pulp.value(problem.objective)) if problem.objective is not None else 0


def core_tu_simple_old(
    vertices: List[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    max_coal_size: int = 3,
    solver: str = "GUROBI",
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    altruist_edges: Optional[Mapping[VertexId, Sequence[VertexId]]] = None,
) -> Dict[str, object]:
    """
    TU-core algorithm with altruists present from the start.
    Objective: lexicographically maximise real patients, then minimise altruist-edge usage
    via big-M weighting. Utilities exclude altruists. Coalition constraints ensure stability
    for all coalitions of size ≤ max_coal_size.
    """
    if Delta not in (2, 3):
        raise ValueError("Delta must be 2 or 3")

    rng = rng or np.random.default_rng()

    # Stage A: Baseline without altruists → compute K (max real patients)
    base_cycle_db = enumerate_cycles(vertices, adj_out, partition, Delta)
    base_solution, base_status = _solve_cycle_ip(base_cycle_db, Delta, partition, solver)
    if base_status not in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved):
        return {
            "solution": set(),
            "in_core": False,
            "player_utilities": {},
            "objective_value": 0,
            "objective_real_patients": 0,
            "objective_altruist_penalty": 0,
            "num_coalitions": 0,
            "max_coal_size": max_coal_size,
            "altruists_present": bool(altruist_edges),
            "M": 0,
            "altruists_used": 0,
        }
    K = sum(base_cycle_db.cycles[cid].length for cid in base_solution)

    # Stage B: Build full graph with at most 5% altruists and enforce coalition constraints
    limited_altruists = {}
    if altruist_edges:
        max_allowed = int(math.floor(0.05 * len(vertices)))
        if max_allowed > 0:
            selected_keys = sorted(altruist_edges.keys())[:max_allowed]
            limited_altruists = {k: altruist_edges[k] for k in selected_keys}
    full_vertices, full_adj, altruists = _augment_with_altruists(vertices, adj_out, limited_altruists, rng=rng)

    # Enumerate cycles (with altruists) under Delta
    cycle_db = enumerate_cycles(full_vertices, full_adj, partition, Delta, altruists)
    cycles = [c for c in cycle_db.cycles if c.length <= Delta]

    # Compute coalition bounds opt_S
    players = partition.players
    coalition_bounds: List[Tuple[Set[PlayerId], int]] = []
    for size in range(1, min(max_coal_size, len(players)) + 1):
        for coalition_tuple in combinations(players, size):
            S = set(coalition_tuple)
            opt_S = _compute_opt_for_coalition(
                cycle_db, partition, S, Delta, solver, time_limit=time_limit, mip_gap=mip_gap
            )
            coalition_bounds.append((S, opt_S))

    # Main IP: Minimize altruist-edge incidence subject to coverage ≥ K and coalition cuts
    problem = pulp.LpProblem("TUCoreSimple", pulp.LpMinimize)
    y_vars = {c.id: pulp.LpVariable(f"x_{c.id}", lowBound=0, upBound=1, cat="Binary") for c in cycles}

    # Precompute objective parts per cycle
    real_terms = []
    altruist_pen_terms = []
    for c in cycles:
        real_count = c.non_altruist_count
        alt_pen, _ = _altruist_incidence_stats(c, set(altruists))
        real_terms.append(y_vars[c.id] * real_count)
        altruist_pen_terms.append(y_vars[c.id] * alt_pen)
    # Objective: minimize total altruist-edge incidence
    problem += pulp.lpSum(altruist_pen_terms)

    # Vertex disjointness for all vertices in the cycle DB (altruists included)
    for vertex, cids in cycle_db.by_vertex.items():
        relevant = [y_vars[cid] for cid in cids if cid in y_vars]
        if relevant:
            problem += pulp.lpSum(relevant) <= 1, f"disjoint_{vertex}"

    # Coverage lower bound: meet at least baseline K real patients
    if real_terms:
        problem += pulp.lpSum(real_terms) >= int(K), "coverage_lower_bound"

    # Coalition constraints: Σ_{c} x_c · (Σ_{i∈S} α_{c,i}) ≥ opt_S
    for idx, (S, bound) in enumerate(coalition_bounds):
        if bound <= 0:
            continue
        lhs_terms = []
        for c in cycles:
            weight = sum(c.player_counts.get(player, 0) for player in S)
            if weight:
                lhs_terms.append(y_vars[c.id] * weight)
        if lhs_terms:
            problem += pulp.lpSum(lhs_terms) >= int(bound), f"coalition_{idx}"

    solver_instance = make_pulp_solver(solver, time_limit=time_limit, mip_gap=mip_gap)
    problem.solve(solver_instance)

    status = problem.status
    selected: Set[int] = set(cid for cid, var in y_vars.items() if var.value() is not None and var.value() > 0.5)

    if status not in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved):
        return {
            "solution": selected,
            "in_core": False,
            "player_utilities": {},
            "objective_value": 0,
            "objective_real_patients": 0,
            "objective_altruist_penalty": 0,
            "num_coalitions": len(coalition_bounds),
            "max_coal_size": max_coal_size,
            "altruists_present": bool(altruists),
            "M": len(set(full_vertices)),
            "altruists_used": 0,
        }

    # Stats
    player_utilities = compute_player_utilities(selected, cycle_db)

    def _safe_value(expr) -> float:
        raw = pulp.value(expr) if expr is not None else None
        return float(raw) if raw is not None else 0.0

    objective_value = int(round(_safe_value(problem.objective)))
    objective_real_patients = int(round(_safe_value(pulp.lpSum(real_terms)))) if real_terms else 0
    objective_altruist_penalty = int(round(_safe_value(pulp.lpSum(altruist_pen_terms)))) if altruist_pen_terms else 0

    selected_altruists: Set[int] = set()
    for cid in selected:
        cyc = cycle_db.cycles[cid]
        for v in cyc.vertices:
            if v in altruists:
                selected_altruists.add(v)

    return {
        "solution": selected,
        "in_core": True if status in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved) else False,
        "player_utilities": player_utilities,
        "objective_value": objective_value,
        "objective_real_patients": objective_real_patients,
        "objective_altruist_penalty": objective_altruist_penalty,
        "num_coalitions": len(coalition_bounds),
        "max_coal_size": max_coal_size,
        "altruists_added": bool(altruists),
        "M": len(set(full_vertices)),
        "altruists_used": len(selected_altruists),
    }





def core_tu_simple(
    vertices: List[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    max_coal_size: int = 3,
    solver: str = "GUROBI",
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    altruist_edges: Optional[Mapping[VertexId, Sequence[VertexId]]] = None,
) -> Dict[str, object]:
    """
    TU-core algorithm that incrementally augments the instance with altruists.
    We first solve the max matching IP without altruists to obtain the baseline
    coverage. Coalition cuts (size ≤ max_coal_size) are added alongside this
    coverage constraint. If the resulting polytope is infeasible, we add one
    randomly-chosen altruist (if available) and try again, reusing the same cuts.
    """
    if Delta not in (2, 3):
        raise ValueError("Delta must be 2 or 3")

    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}

    # Baseline solve without altruists to determine minimum real transplants.
    base_cycle_db = enumerate_cycles(vertices, adj_out, partition, Delta)
    base_solution, base_status = _solve_cycle_ip(base_cycle_db, Delta, partition, solver)
    if not (
        base_status == pulp.LpStatusOptimal
        or (base_status == pulp.LpStatusNotSolved and base_solution)
    ):
        return {
            "solution": set(),
            "in_core": False,
            "player_utilities": {},
            "objective_value": 0,
            "objective_real_patients": 0,
            "objective_altruist_penalty": 0,
            "num_coalitions": 0,
            "max_coal_size": max_coal_size,
            "altruists_present": False,
            "M": len(set(vertices)),
            "altruists_used": 0,
        }
    min_real_transplants = sum(
        base_cycle_db.cycles[cid].non_altruist_count for cid in base_solution
    )

    # Coalition bounds remain valid regardless of altruists (only non-altruist cycles count).
    players = partition.players
    coalition_bounds: List[Tuple[Set[PlayerId], int]] = []
    for size in range(1, min(max_coal_size, len(players)) + 1):
        for coalition_tuple in combinations(players, size):
            S = set(coalition_tuple)
            opt_S = _compute_opt_for_coalition(
                base_cycle_db,
                partition,
                S,
                Delta,
                solver,
                time_limit=time_limit,
                mip_gap=mip_gap,
            )
            coalition_bounds.append((S, opt_S))

    # Prepare mutable graph copies and the pool of altruists that can be added.
    working_adj = {u: list(neigh) for u, neigh in adj_out.items()}
    current_vertices = list(vertices)
    altruists: List[VertexId] = []

    existing_ids = set(vertices) | set(adj_out.keys())
    available_altruists = [
        altruist for altruist in altruist_edges if altruist not in existing_ids
    ]
    rng.shuffle(available_altruists)

    def _build_and_solve(
        cycle_db: CycleDB, altruist_set: Set[VertexId]
    ) -> Tuple[pulp.LpProblem, int, Set[CycleId], Dict[CycleId, int], Dict[CycleId, int]]:
        cycles = [c for c in cycle_db.cycles if c.length <= Delta]
        problem = pulp.LpProblem("TUCoreSimpleIter", pulp.LpMinimize)
        y_vars = {
            c.id: pulp.LpVariable(f"x_{c.id}", lowBound=0, upBound=1, cat="Binary")
            for c in cycles
        }

        real_counts: Dict[CycleId, int] = {}
        penalty_counts: Dict[CycleId, int] = {}
        real_terms = []
        altruist_pen_terms = []
        for c in cycles:
            real_count = c.non_altruist_count
            penalty, _ = _altruist_incidence_stats(c, altruist_set)
            real_counts[c.id] = real_count
            penalty_counts[c.id] = penalty
            real_terms.append(y_vars[c.id] * real_count)
            altruist_pen_terms.append(y_vars[c.id] * penalty)
        problem += pulp.lpSum(altruist_pen_terms)

        for vertex, cids in cycle_db.by_vertex.items():
            relevant = [y_vars[cid] for cid in cids if cid in y_vars]
            if relevant:
                problem += pulp.lpSum(relevant) <= 1, f"disjoint_{vertex}"

        if real_terms and min_real_transplants > 0:
            problem += pulp.lpSum(real_terms) >= int(min_real_transplants), "coverage_lower_bound"

        for idx, (S, bound) in enumerate(coalition_bounds):
            if bound <= 0:
                continue
            lhs_terms = []
            for c in cycles:
                weight = sum(c.player_counts.get(player, 0) for player in S)
                if weight:
                    lhs_terms.append(y_vars[c.id] * weight)
            if lhs_terms:
                problem += pulp.lpSum(lhs_terms) >= int(bound), f"coalition_{idx}"

        solver_instance = make_pulp_solver(solver, time_limit=time_limit, mip_gap=mip_gap)
        problem.solve(solver_instance)
        status = problem.status
        selected = {
            cid for cid, var in y_vars.items() if var.value() is not None and var.value() > 0.5
        }
        return problem, status, selected, real_counts, penalty_counts

    def _safe_value(expr) -> float:
        raw = pulp.value(expr) if expr is not None else None
        return float(raw) if raw is not None else 0.0

    final_result: Optional[Dict[str, object]] = None
    final_cycle_db: Optional[CycleDB] = None

    while True:
        altruist_set = set(altruists)
        cycle_db = enumerate_cycles(current_vertices, working_adj, partition, Delta, altruists)
        problem, status, selected, real_counts, penalty_counts = _build_and_solve(
            cycle_db, altruist_set
        )

        feasible = status == pulp.LpStatusOptimal or (
            status == pulp.LpStatusNotSolved and bool(selected)
        )
        if feasible:
            player_utilities = compute_player_utilities(selected, cycle_db)
            objective_value = int(round(_safe_value(problem.objective)))
            objective_real_patients = sum(real_counts.get(cid, 0) for cid in selected)
            objective_altruist_penalty = sum(penalty_counts.get(cid, 0) for cid in selected)


            final_result = {
                "solution": selected,
                "in_core": True,
                "player_utilities": player_utilities,
                "objective_value": objective_value,
                "objective_real_patients": objective_real_patients,
                "objective_altruist_penalty": objective_altruist_penalty,
                "num_coalitions": len(coalition_bounds),
                "max_coal_size": max_coal_size,
                "altruists_present": bool(altruists),
                "M": len(set(current_vertices)),
                "altruists_used": len(altruists),
            }
            final_cycle_db = cycle_db
            break

        if status == pulp.LpStatusInfeasible and available_altruists:
            new_index = int(rng.integers(len(available_altruists)))
            new_altruist = available_altruists.pop(new_index)
            targets = [
                t for t in altruist_edges.get(new_altruist, []) if t in current_vertices
            ]
            _add_altruist_vertex(
                working_adj,
                current_vertices,
                new_altruist,
                targets,
                rng=rng,
            )
            altruists.append(new_altruist)
            current_vertices.append(new_altruist)
            continue

        # Either infeasible with no altruists left or solver returned a fatal status.
        final_result = {
            "solution": set(),
            "in_core": False,
            "player_utilities": {},
            "objective_value": 0,
            "objective_real_patients": 0,
            "objective_altruist_penalty": 0,
            "num_coalitions": len(coalition_bounds),
            "max_coal_size": max_coal_size,
            "altruists_present": bool(altruists),
            "M": len(set(current_vertices)),
            "altruists_used": 0,
        }
        final_cycle_db = cycle_db
        break

    return final_result
