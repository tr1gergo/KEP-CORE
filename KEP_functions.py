from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
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


def make_pulp_solver(
    solver: str,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    msg: bool = False,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> pulp.LpSolver:
    """Return a configured PuLP solver."""
    solver_name = (solver or "CBC").upper()
    normalized_seed = (
        None
        if solver_seed is None
        else 1 + int(solver_seed) % 1_999_999_999
    )
    if solver_name == "GUROBI":
        try:
            solver_params: Dict[str, object] = {}
            if threads is not None:
                solver_params["Threads"] = int(threads)
            if normalized_seed is not None:
                solver_params["Seed"] = normalized_seed
            return pulp.GUROBI(
                msg=msg,
                timeLimit=time_limit,
                gapRel=mip_gap,
                **solver_params,
            )
        except pulp.PulpSolverError:
            pass
    cbc_options = []
    if normalized_seed is not None:
        cbc_options.append(f"randomSeed={normalized_seed}")
    return pulp.PULP_CBC_CMD(
        msg=msg,
        timeLimit=time_limit,
        gapRel=mip_gap,
        threads=threads,
        options=cbc_options or None,
    )


def _safe_solver_attribute(model: object, name: str) -> Optional[float]:
    """Read an optional solver attribute without coupling the code to Gurobi."""
    if model is None:
        return None
    try:
        value = getattr(model, name)
    except (AttributeError, TypeError):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _solve_with_diagnostics(
    problem: pulp.LpProblem,
    solver_instance: pulp.LpSolver,
) -> Dict[str, object]:
    """Solve a PuLP model and return portable runtime/status diagnostics."""
    started = perf_counter()
    problem.solve(solver_instance)
    elapsed = perf_counter() - started
    solver_model = getattr(problem, "solverModel", None)
    return {
        "status_code": int(problem.status),
        "status": pulp.LpStatus.get(problem.status, str(problem.status)),
        "runtime_seconds": float(elapsed),
        "solver_runtime_seconds": _safe_solver_attribute(solver_model, "Runtime"),
        "mip_gap": _safe_solver_attribute(solver_model, "MIPGap"),
        "node_count": _safe_solver_attribute(solver_model, "NodeCount"),
        "solution_count": _safe_solver_attribute(solver_model, "SolCount"),
    }


def _cycle_weight(cycle: Cycle, has_altruist_mode: bool) -> int:
    return cycle.non_altruist_count if has_altruist_mode else cycle.length


def _normalize_lexicographic_tier_value(name: str, value: object) -> float:
    """Remove solver noise from lexicographic tiers with integer coefficients."""
    numeric = float(value or 0.0)
    if name in {"transplants", "cycle_count", "same_blood"}:
        return float(round(numeric))
    return numeric


def _solve_cycle_ip(
    cycle_db: CycleDB,
    Delta: int,
    partition: Partition,
    solver: str,
    cuts: Optional[List[Dict[str, object]]] = None,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
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
            problem += pulp.lpSum(lhs) >= rhs, f"cut_{idx}"

    solver_instance = make_pulp_solver(
        solver,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        solver_seed=solver_seed,
    )
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
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
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
    # Enforce disjointness for every vertex represented in the cycle database,
    # including introduced altruists.  Restricting this loop to owned patient-
    # donor pairs would allow the same altruist to appear in several selected
    # donor-started chains.
    for vertex in cycle_db.by_vertex.keys():
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
            threads=threads,
            solver_seed=solver_seed,
        )
        problem.solve(solver_instance)
        status = problem.status
        raw_objective_value = (
            pulp.value(expr)
            if problem.status in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved)
            else 0.0
        )
        objective_values[stage_name] = _normalize_lexicographic_tier_value(
            stage_name, raw_objective_value
        )
        # A time-limited incumbent is useful diagnostically but does not certify
        # a lexicographic tier.  Stop instead of fixing an unproven objective and
        # later mislabeling the four-stage result as optimal.
        if status != pulp.LpStatusOptimal:
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
    donor_order: Optional[Sequence[VertexId]] = None,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = 0.0,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Iteratively apply lexicographic optimization, adding altruists only when needed for core feasibility.
    """
    run_started = perf_counter()
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}
    working_adj = {u: list(neigh) for u, neigh in adj_out.items()}
    edge_same_blood = compute_edge_same_blood(working_adj, graph_features)
    current_vertices = list(vertices)
    altruists_added: List[VertexId] = []
    available_altruists = _ordered_altruists(
        altruist_edges,
        set(working_adj),
        donor_order,
        rng,
    )
    full_donor_order = list(available_altruists)
    warm_start: Optional[Set[CycleId]] = None
    history: List[Dict[str, object]] = []
    cycle_enumeration_seconds = 0.0
    optimization_seconds = 0.0
    separation_seconds = 0.0
    optimization_solves = 0
    separation_solves = 0

    def finish(
        selection: Set[CycleId],
        cycle_db: CycleDB,
        objectives: Dict[str, float],
        *,
        in_core: bool,
        certified: bool,
        reason: str,
        status: int,
    ) -> Dict[str, object]:
        used = _used_altruists(selection, cycle_db, altruists_added)
        return {
            "solution": selection,
            "altruists_added": len(altruists_added),
            "altruists_introduced": len(altruists_added),
            "altruists_used": len(used),
            "introduced_donor_ids": list(altruists_added),
            "used_donor_ids": sorted(used),
            "donor_order": full_donor_order,
            "final_in_core": bool(in_core),
            "certified": bool(certified),
            "termination_reason": reason,
            "solver_status": pulp.LpStatus.get(status, str(status)),
            "objective_tiers": objectives,
            "player_utilities": compute_player_utilities(selection, cycle_db),
            "runtime_seconds": perf_counter() - run_started,
            "cycle_enumeration_seconds": cycle_enumeration_seconds,
            "optimization_seconds": optimization_seconds,
            "separation_seconds": separation_seconds,
            "num_optimization_solves": optimization_solves,
            "num_separation_solves": separation_solves,
            "history": history,
        }

    while True:
        vertex_hardness = compute_vertex_hardness(working_adj)
        enumeration_started = perf_counter()
        cycle_db = enumerate_cycles(
            current_vertices,
            working_adj,
            partition,
            Delta,
            altruists_added,
            edge_same_blood=edge_same_blood,
            vertex_hardness=vertex_hardness,
        )
        cycle_enumeration_seconds += perf_counter() - enumeration_started
        optimization_started = perf_counter()
        selection, objectives, status = solve_lexicographic_cycle_cover(
            cycle_db,
            Delta,
            partition,
            solver=solver,
            warm_start=warm_start,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        optimization_seconds += perf_counter() - optimization_started
        optimization_solves += 4
        history.append(
            {
                "donors_introduced": len(altruists_added),
                "phase": "lexicographic_optimization",
                "status": pulp.LpStatus.get(status, str(status)),
                "objectives": objectives,
                "solution_size": len(selection),
            }
        )
        if status != pulp.LpStatusOptimal:
            return finish(
                selection,
                cycle_db,
                objectives,
                in_core=False,
                certified=False,
                reason="lexicographic_solve_not_optimal",
                status=status,
            )

        separation_started = perf_counter()
        separation = separate_blocking_coalition(
            selection,
            cycle_db,
            partition,
            max_coal_size,
            Delta,
            core_type="weak",
            solver=solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        separation_seconds += perf_counter() - separation_started
        separation_solves += 1
        history.append(
            {
                "donors_introduced": len(altruists_added),
                "phase": "weak_core_separation",
                "status": separation["diagnostics"]["status"],
                "blocking": separation["blocking"],
                "coalition_size": len(separation["coalition"]),
            }
        )
        if separation["blocking"] is False and separation["certified"]:
            return finish(
                selection,
                cycle_db,
                objectives,
                in_core=True,
                certified=True,
                reason="stable",
                status=status,
            )
        if separation["blocking"] is None:
            return finish(
                selection,
                cycle_db,
                objectives,
                in_core=False,
                certified=False,
                reason="separation_not_certified",
                status=int(separation["diagnostics"]["status_code"]),
            )

        if not available_altruists or (
            max_added_altruists is not None and len(altruists_added) >= max_added_altruists
        ):
            reason = "unstable_at_donor_limit" if available_altruists else "unstable_after_all_donors"
            return finish(
                selection,
                cycle_db,
                objectives,
                in_core=False,
                certified=True,
                reason=reason,
                status=status,
            )

        new_altruist = available_altruists.pop(0)
        targets = [
            int(target)
            for target in altruist_edges.get(new_altruist, [])
            if int(target) in partition.owner_of
        ]
        _add_altruist_vertex(
            working_adj,
            current_vertices,
            new_altruist,
            targets,
            edge_same_blood=edge_same_blood,
            graph_features=graph_features,
            rng=rng,
            real_vertices=set(partition.owner_of),
        )
        altruists_added.append(new_altruist)
        current_vertices.append(new_altruist)
        warm_start = selection


def prepare_lexicographic_floor_baseline(
    vertices: List[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    graph_features: GraphFeatures,
    max_coal_size: int = 3,
    solver: str = "GUROBI",
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = 0.0,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Solve and audit the donor-free lexicographic benchmark once.

    The returned selection and four optimal tier values can be reused across
    donor orders by :func:`lexicographic_floor_core_search`.  Recipient
    hardness is measured on the donor-free graph and is deliberately frozen,
    so the fourth-tier floor keeps the same meaning after donor arcs are added.
    """
    run_started = perf_counter()
    edge_same_blood = compute_edge_same_blood(adj_out, graph_features)
    vertex_hardness = compute_vertex_hardness(adj_out)
    enumeration_started = perf_counter()
    cycle_db = enumerate_cycles(
        vertices,
        adj_out,
        partition,
        Delta,
        edge_same_blood=edge_same_blood,
        vertex_hardness=vertex_hardness,
    )
    cycle_enumeration_seconds = perf_counter() - enumeration_started
    optimization_started = perf_counter()
    selection, objective_tiers, status = solve_lexicographic_cycle_cover(
        cycle_db,
        Delta,
        partition,
        solver=solver,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        solver_seed=solver_seed,
    )
    optimization_seconds = perf_counter() - optimization_started
    separation_seconds = 0.0
    separation: Optional[Dict[str, object]] = None
    initial_cut: Optional[Dict[str, object]] = None

    if status == pulp.LpStatusOptimal:
        separation_started = perf_counter()
        separation = separate_blocking_coalition(
            selection,
            cycle_db,
            partition,
            max_coal_size,
            Delta,
            core_type="weak",
            solver=solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        separation_seconds = perf_counter() - separation_started
        if separation["blocking"] is True:
            coalition = tuple(sorted(int(i) for i in separation["coalition"]))
            baseline_utilities = separation["baseline_utilities"]
            initial_cut = {
                "coalition": coalition,
                "rhs": int(sum(baseline_utilities.get(i, 0) for i in coalition) + 1),
            }

    certified = bool(
        status == pulp.LpStatusOptimal
        and separation is not None
        and separation["certified"]
        and separation["blocking"] is not None
    )
    in_core = bool(
        certified and separation is not None and separation["blocking"] is False
    )
    return {
        "selection": selection,
        "objective_tiers": dict(objective_tiers),
        "solver_status_code": int(status),
        "solver_status": pulp.LpStatus.get(status, str(status)),
        "certified": certified,
        "in_core": in_core,
        "separation": separation,
        "initial_cut": initial_cut,
        "vertex_hardness": vertex_hardness,
        "runtime_seconds": perf_counter() - run_started,
        "cycle_enumeration_seconds": cycle_enumeration_seconds,
        "optimization_seconds": optimization_seconds,
        "separation_seconds": separation_seconds,
        "num_optimization_solves": len(objective_tiers),
        "num_separation_solves": int(separation is not None),
    }


def lexicographic_floor_core_search(
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
    donor_order: Optional[Sequence[VertexId]] = None,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = 0.0,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
    baseline: Optional[Mapping[str, object]] = None,
    floor_tolerance: float = 1e-7,
) -> Dict[str, object]:
    """Stabilize while preserving donor-free lexicographic objective floors.

    First, the donor-free four-tier lexicographic optimum is solved and
    exactly audited (or a prepared ``baseline`` is reused).  Its four values
    become lower bounds rather than objectives.  The master then minimizes
    donors used and adds aggregate weak-core search cuts.  A donor is
    introduced only after this cut-augmented floor master is infeasible.

    Every successful return is exactly weak-core verified.  The aggregate
    cuts are heuristic for weak-core *existence*, however, so an unsuccessful
    return does not certify that no stable allocation satisfies the floors.
    """
    if Delta not in (2, 3):
        raise ValueError("Delta must be 2 or 3")
    if max_coal_size <= 0:
        raise ValueError("max_coal_size must be positive")
    if floor_tolerance < 0:
        raise ValueError("floor_tolerance must be nonnegative")

    run_started = perf_counter()
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}
    real_vertices = set(int(v) for v in vertices)
    ordered_donors = _ordered_altruists(
        altruist_edges,
        real_vertices.union(int(v) for v in adj_out),
        donor_order,
        rng,
    )
    baseline_was_prepared = baseline is not None
    if baseline is None:
        baseline = prepare_lexicographic_floor_baseline(
            vertices,
            adj_out,
            partition,
            Delta,
            graph_features,
            max_coal_size=max_coal_size,
            solver=solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )

    required_tiers = ("transplants", "cycle_count", "same_blood", "hard_match")
    objective_floors = {
        name: _normalize_lexicographic_tier_value(
            name, dict(baseline.get("objective_tiers", {})).get(name, 0.0)
        )
        for name in required_tiers
    }
    missing_tiers = [
        name for name in required_tiers if name not in dict(baseline.get("objective_tiers", {}))
    ]
    if missing_tiers:
        raise ValueError(f"baseline is missing objective tiers: {missing_tiers}")

    working_adj = {int(u): list(neighbors) for u, neighbors in adj_out.items()}
    edge_same_blood = compute_edge_same_blood(working_adj, graph_features)
    # Freeze the fourth-tier score on the donor-free compatibility graph.  If
    # donor arcs were allowed to change recipient indegrees, the recorded
    # donor-free floor would not be comparable to an augmented allocation.
    vertex_hardness = {
        int(vertex): float(value)
        for vertex, value in dict(
            baseline.get("vertex_hardness", compute_vertex_hardness(adj_out))
        ).items()
    }
    current_vertices = list(vertices)
    introduced: List[VertexId] = []
    remaining_donors = list(ordered_donors)
    cuts: List[Dict[str, object]] = []
    cut_keys: Set[Tuple[Tuple[PlayerId, ...], int]] = set()
    initial_cut = baseline.get("initial_cut")
    if initial_cut is not None:
        coalition = tuple(sorted(int(i) for i in initial_cut["coalition"]))
        rhs = int(initial_cut["rhs"])
        cuts.append({"coalition": coalition, "rhs": rhs})
        cut_keys.add((coalition, rhs))

    cycle_enumeration_seconds = 0.0
    optimization_seconds = 0.0
    separation_seconds = 0.0
    master_solves = 0
    separation_solves = 0
    solve_gaps: List[float] = []
    history: List[Dict[str, object]] = []

    def enumerate_current() -> CycleDB:
        nonlocal cycle_enumeration_seconds
        started = perf_counter()
        database = enumerate_cycles(
            current_vertices,
            working_adj,
            partition,
            Delta,
            introduced,
            edge_same_blood=edge_same_blood,
            vertex_hardness=vertex_hardness,
        )
        cycle_enumeration_seconds += perf_counter() - started
        return database

    def tier_values(selection: Set[CycleId], cycle_db: CycleDB) -> Dict[str, float]:
        chosen = [cycle_db.cycles[cid] for cid in selection]
        return {
            "transplants": float(sum(c.non_altruist_count for c in chosen)),
            "cycle_count": float(len(chosen)),
            "same_blood": float(sum(c.same_blood_edges for c in chosen)),
            "hard_match": float(sum(c.hard_match_score for c in chosen)),
        }

    def finish(
        *,
        selection: Set[CycleId],
        cycle_db: CycleDB,
        in_core: bool,
        certified: bool,
        reason: str,
        solver_status: str,
    ) -> Dict[str, object]:
        used = _used_altruists(selection, cycle_db, introduced)
        achieved = tier_values(selection, cycle_db)
        return {
            "solution": selection,
            "in_core": bool(in_core),
            "final_in_core": bool(in_core),
            "certified": bool(certified),
            "termination_reason": reason,
            "solver_status": solver_status,
            "player_utilities": compute_player_utilities(selection, cycle_db),
            "objective_value": len(used),
            "objective_real_patients": int(round(achieved["transplants"])),
            "objective_tiers": achieved,
            "baseline_objective_tiers": dict(objective_floors),
            "objective_floor_slacks": {
                name: achieved[name] - objective_floors[name]
                for name in required_tiers
            },
            "baseline_real_transplants": int(round(objective_floors["transplants"])),
            "objective_altruist_penalty": len(used),
            "altruists_introduced": len(introduced),
            "altruists_added": len(introduced),
            "altruists_used": len(used),
            "introduced_donor_ids": list(introduced),
            "used_donor_ids": sorted(used),
            "donor_order": list(ordered_donors),
            "num_cuts": len(cuts),
            "num_master_solves": master_solves,
            "num_optimization_solves": master_solves,
            "num_separation_solves": separation_solves,
            "runtime_seconds": perf_counter() - run_started,
            "cycle_enumeration_seconds": cycle_enumeration_seconds,
            "optimization_seconds": optimization_seconds,
            "separation_seconds": separation_seconds,
            "max_mip_gap": max(solve_gaps) if solve_gaps else 0.0,
            "baseline_reused": bool(baseline_was_prepared),
            "baseline_preparation_seconds": float(baseline.get("runtime_seconds", 0.0)),
            "objective_floor_type": "donor_free_lexicographic_lower_bounds",
            "objective_score_reference": "donor_free_graph",
            "search_cut_type": "aggregate_weak_core_heuristic",
            "history": history,
        }

    baseline_status = int(baseline.get("solver_status_code", pulp.LpStatusUndefined))
    if not baseline.get("certified"):
        base_cycle_db = enumerate_current()
        return finish(
            selection=set(baseline.get("selection", set())),
            cycle_db=base_cycle_db,
            in_core=False,
            certified=False,
            reason="baseline_lexicographic_or_separation_not_certified",
            solver_status=str(baseline.get("solver_status", "Unknown")),
        )
    if baseline.get("in_core"):
        base_cycle_db = enumerate_current()
        return finish(
            selection=set(baseline.get("selection", set())),
            cycle_db=base_cycle_db,
            in_core=True,
            certified=True,
            reason="stable_donor_free_lexicographic_baseline",
            solver_status=pulp.LpStatus.get(baseline_status, str(baseline_status)),
        )
    if not cuts:
        base_cycle_db = enumerate_current()
        return finish(
            selection=set(baseline.get("selection", set())),
            cycle_db=base_cycle_db,
            in_core=False,
            certified=False,
            reason="blocked_baseline_missing_initial_cut",
            solver_status=pulp.LpStatus.get(baseline_status, str(baseline_status)),
        )

    while True:
        cycle_db = enumerate_current()
        cycles = [cycle for cycle in cycle_db.cycles if cycle.length <= Delta]
        problem = pulp.LpProblem("LexicographicFloorStabilization", pulp.LpMinimize)
        x_vars = {
            cycle.id: pulp.LpVariable(
                f"lex_floor_{cycle.id}", lowBound=0, upBound=1, cat="Binary"
            )
            for cycle in cycles
        }
        problem += pulp.lpSum(
            x_vars[cycle.id] * cycle.altruist_count for cycle in cycles
        )
        for vertex, cycle_ids in cycle_db.by_vertex.items():
            relevant = [x_vars[cid] for cid in cycle_ids if cid in x_vars]
            if relevant:
                problem += pulp.lpSum(relevant) <= 1, f"lex_floor_disjoint_{vertex}"

        tier_expressions = {
            "transplants": pulp.lpSum(
                x_vars[c.id] * c.non_altruist_count for c in cycles
            ),
            "cycle_count": pulp.lpSum(x_vars[c.id] for c in cycles),
            "same_blood": pulp.lpSum(
                x_vars[c.id] * c.same_blood_edges for c in cycles
            ),
            "hard_match": pulp.lpSum(
                x_vars[c.id] * c.hard_match_score for c in cycles
            ),
        }
        for name in required_tiers:
            problem += (
                tier_expressions[name] >= objective_floors[name] - floor_tolerance
            ), f"lex_floor_{name}"

        for cut_index, cut in enumerate(cuts):
            coalition = set(int(i) for i in cut["coalition"])
            lhs = pulp.lpSum(
                x_vars[cycle.id]
                * sum(cycle.player_counts.get(i, 0) for i in coalition)
                for cycle in cycles
            )
            problem += lhs >= int(cut["rhs"]), f"lex_weak_cut_{cut_index}"

        solver_instance = make_pulp_solver(
            solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        diagnostics = _solve_with_diagnostics(problem, solver_instance)
        optimization_seconds += float(diagnostics["runtime_seconds"])
        master_solves += 1
        if diagnostics.get("mip_gap") is not None:
            solve_gaps.append(float(diagnostics["mip_gap"]))
        selection = {
            cycle_id
            for cycle_id, variable in x_vars.items()
            if variable.value() is not None and variable.value() > 0.5
        }
        history.append(
            {
                "donors_introduced": len(introduced),
                "phase": "lexicographic_floor_master",
                "status": diagnostics["status"],
                "runtime_seconds": diagnostics["runtime_seconds"],
                "num_cuts": len(cuts),
            }
        )

        if problem.status == pulp.LpStatusInfeasible:
            donor_limit_reached = (
                not remaining_donors
                or (
                    max_added_altruists is not None
                    and len(introduced) >= max_added_altruists
                )
            )
            if donor_limit_reached:
                return finish(
                    selection=set(),
                    cycle_db=cycle_db,
                    in_core=False,
                    certified=False,
                    reason="floor_master_infeasible_at_donor_limit",
                    solver_status=str(diagnostics["status"]),
                )
            new_altruist = remaining_donors.pop(0)
            targets = [
                int(target)
                for target in altruist_edges.get(new_altruist, [])
                if int(target) in real_vertices
            ]
            _add_altruist_vertex(
                working_adj,
                current_vertices,
                new_altruist,
                targets,
                edge_same_blood=edge_same_blood,
                graph_features=graph_features,
                rng=rng,
                real_vertices=real_vertices,
            )
            introduced.append(new_altruist)
            current_vertices.append(new_altruist)
            continue

        if problem.status != pulp.LpStatusOptimal:
            return finish(
                selection=selection,
                cycle_db=cycle_db,
                in_core=False,
                certified=False,
                reason="floor_master_not_optimal",
                solver_status=str(diagnostics["status"]),
            )

        achieved = tier_values(selection, cycle_db)
        if any(
            achieved[name] + floor_tolerance < objective_floors[name]
            for name in required_tiers
        ):
            return finish(
                selection=selection,
                cycle_db=cycle_db,
                in_core=False,
                certified=False,
                reason="objective_floor_validation_failed",
                solver_status=str(diagnostics["status"]),
            )

        separation_started = perf_counter()
        separation = separate_blocking_coalition(
            selection,
            cycle_db,
            partition,
            max_coal_size,
            Delta,
            core_type="weak",
            solver=solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        separation_seconds += perf_counter() - separation_started
        separation_solves += 1
        separation_diag = separation["diagnostics"]
        if separation_diag.get("mip_gap") is not None:
            solve_gaps.append(float(separation_diag["mip_gap"]))
        history.append(
            {
                "donors_introduced": len(introduced),
                "phase": "weak_core_separation",
                "status": separation_diag["status"],
                "runtime_seconds": separation_diag["runtime_seconds"],
                "blocking": separation["blocking"],
                "coalition_size": len(separation["coalition"]),
            }
        )

        if separation["blocking"] is False and separation["certified"]:
            return finish(
                selection=selection,
                cycle_db=cycle_db,
                in_core=True,
                certified=True,
                reason="stable_with_lexicographic_floors",
                solver_status=str(diagnostics["status"]),
            )
        if separation["blocking"] is None:
            return finish(
                selection=selection,
                cycle_db=cycle_db,
                in_core=False,
                certified=False,
                reason="separation_not_certified",
                solver_status=str(separation_diag["status"]),
            )

        coalition = tuple(sorted(int(i) for i in separation["coalition"]))
        baseline_utilities = separation["baseline_utilities"]
        rhs = int(sum(baseline_utilities.get(i, 0) for i in coalition) + 1)
        cut_key = (coalition, rhs)
        if cut_key in cut_keys:
            return finish(
                selection=selection,
                cycle_db=cycle_db,
                in_core=False,
                certified=False,
                reason="duplicate_violated_weak_cut",
                solver_status=str(separation_diag["status"]),
            )
        cut_keys.add(cut_key)
        cuts.append({"coalition": coalition, "rhs": rhs})


def separate_blocking_coalition(
    solution: Set[CycleId],
    cycle_db: CycleDB,
    partition: Partition,
    max_coal_size: int,
    Delta: int,
    core_type: str = "weak",
    solver: str = "GUROBI",
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Find one capped blocking coalition in a single exact separation MIP.

    ``core_type`` may be ``"weak"`` (all members strictly improve),
    ``"strong"`` (all weakly improve and at least one strictly improves), or
    ``"tu"`` (the coalition's total deviation value exceeds its allocation
    utility).  Only donor-free cycles are available to a deviating coalition.

    A result with ``certified=True`` and ``blocking=False`` certifies stability
    against every coalition of size at most ``max_coal_size``.  An interrupted
    solve is returned as uncertified instead of being treated as stability.
    """
    if max_coal_size <= 0:
        raise ValueError("max_coal_size must be positive")
    core_key = str(core_type).lower()
    if core_key not in {"weak", "strong", "tu"}:
        raise ValueError("core_type must be 'weak', 'strong', or 'tu'")

    players = list(partition.players)
    coalition_cap = min(int(max_coal_size), len(players))
    baseline = {player: 0 for player in players}
    baseline.update(compute_player_utilities(solution, cycle_db))
    candidate_cycles = [
        cycle
        for cycle in cycle_db.cycles
        if not cycle.has_altruist and cycle.length <= Delta
    ]
    if coalition_cap <= 0 or not players or not candidate_cycles:
        return {
            "blocking": False,
            "certified": True,
            "coalition": set(),
            "deviation_cycles": set(),
            "deviation_value": 0,
            "deviation_utilities": {},
            "baseline_utilities": baseline,
            "violation": 0,
            "core_type": core_key,
            "diagnostics": {
                "status_code": int(pulp.LpStatusOptimal),
                "status": "NoCycles",
                "runtime_seconds": 0.0,
                "solver_runtime_seconds": 0.0,
                "mip_gap": 0.0,
                "node_count": 0.0,
                "solution_count": 1.0,
            },
        }

    sense = pulp.LpMaximize if core_key == "tu" else pulp.LpMinimize
    problem = pulp.LpProblem(f"{core_key.title()}CoreSeparation", sense)
    z_vars = {
        player: pulp.LpVariable(f"coalition_{player}", lowBound=0, upBound=1, cat="Binary")
        for player in players
    }
    y_vars = {
        cycle.id: pulp.LpVariable(f"deviation_{cycle.id}", lowBound=0, upBound=1, cat="Binary")
        for cycle in candidate_cycles
    }
    coalition_size = pulp.lpSum(z_vars.values())
    problem += coalition_size >= 1, "nonempty_coalition"
    problem += coalition_size <= coalition_cap, "coalition_cap"

    for cycle in candidate_cycles:
        for player in cycle.players_in_cycle:
            problem += y_vars[cycle.id] <= z_vars[player], f"owns_{cycle.id}_{player}"

    candidate_ids = set(y_vars)
    for vertex, cycle_ids in cycle_db.by_vertex.items():
        relevant = [y_vars[cid] for cid in cycle_ids if cid in candidate_ids]
        if relevant:
            problem += pulp.lpSum(relevant) <= 1, f"deviation_disjoint_{vertex}"

    deviation_by_player: Dict[PlayerId, pulp.LpAffineExpression] = {}
    for player in players:
        deviation_by_player[player] = pulp.lpSum(
            y_vars[cycle.id] * cycle.player_counts.get(player, 0)
            for cycle in candidate_cycles
            if cycle.player_counts.get(player, 0)
        )

    if core_key == "weak":
        for player in players:
            problem += (
                deviation_by_player[player]
                >= (baseline.get(player, 0) + 1) * z_vars[player]
            ), f"strict_improvement_{player}"
        problem += coalition_size
    elif core_key == "strong":
        strict_vars = {
            player: pulp.LpVariable(f"strict_{player}", lowBound=0, upBound=1, cat="Binary")
            for player in players
        }
        for player in players:
            problem += strict_vars[player] <= z_vars[player], f"strict_member_{player}"
            problem += (
                deviation_by_player[player]
                >= baseline.get(player, 0) * z_vars[player] + strict_vars[player]
            ), f"weak_improvement_{player}"
        problem += pulp.lpSum(strict_vars.values()) >= 1, "some_strict_improvement"
        problem += coalition_size
    else:
        deviation_total = pulp.lpSum(
            y_vars[cycle.id] * cycle.length for cycle in candidate_cycles
        )
        baseline_total = pulp.lpSum(
            baseline.get(player, 0) * z_vars[player] for player in players
        )
        priority = coalition_cap + 1
        problem += priority * (deviation_total - baseline_total) - coalition_size

    solver_instance = make_pulp_solver(
        solver,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        solver_seed=solver_seed,
    )
    diagnostics = _solve_with_diagnostics(problem, solver_instance)
    status = problem.status

    coalition = {
        player
        for player, variable in z_vars.items()
        if variable.value() is not None and variable.value() > 0.5
    }
    deviation_cycles = {
        cycle_id
        for cycle_id, variable in y_vars.items()
        if variable.value() is not None and variable.value() > 0.5
    }
    deviation_utilities = {
        player: sum(
            cycle_db.cycles[cycle_id].player_counts.get(player, 0)
            for cycle_id in deviation_cycles
        )
        for player in coalition
    }
    deviation_value = sum(
        cycle_db.cycles[cycle_id].length for cycle_id in deviation_cycles
    )
    baseline_total_value = sum(baseline.get(player, 0) for player in coalition)
    violation = deviation_value - baseline_total_value

    if core_key == "tu":
        blocking = bool(coalition and violation > 0)
        certified = status == pulp.LpStatusOptimal
        if not blocking and not certified:
            blocking_value: Optional[bool] = None
        else:
            blocking_value = blocking
    else:
        if status == pulp.LpStatusOptimal:
            blocking_value = True
            certified = True
        elif status == pulp.LpStatusInfeasible:
            blocking_value = False
            certified = True
            coalition = set()
            deviation_cycles = set()
            deviation_utilities = {}
            deviation_value = 0
            violation = 0
        else:
            blocking_value = None
            certified = False

    return {
        "blocking": blocking_value,
        "certified": certified,
        "coalition": coalition,
        "deviation_cycles": deviation_cycles,
        "deviation_value": int(deviation_value),
        "deviation_utilities": deviation_utilities,
        "baseline_utilities": baseline,
        "violation": int(violation),
        "core_type": core_key,
        "diagnostics": diagnostics,
    }


def _add_altruist_vertex(
    adj_out: AdjOut,
    base_vertices: Sequence[VertexId],
    altruist_id: VertexId,
    targets: Sequence[VertexId],
    edge_same_blood: Optional[Dict[Tuple[VertexId, VertexId], int]] = None,
    graph_features: Optional[GraphFeatures] = None,
    rng: Optional[np.random.Generator] = None,
    real_vertices: Optional[Set[VertexId]] = None,
    synthesize_targets_if_empty: bool = False,
) -> VertexId:
    if altruist_id in adj_out:
        raise ValueError(f"Altruist vertex {altruist_id} already present")
    adj_out[altruist_id] = []
    # The return arc closes the cycle representation of a donor-started chain.
    # Other altruists must never receive such an arc: otherwise a cycle can use
    # two altruists and the reported donor count ceases to have its intended
    # operational meaning.
    eligible_return_vertices = (
        [v for v in base_vertices if v in real_vertices]
        if real_vertices is not None
        else list(base_vertices)
    )
    for v in eligible_return_vertices:
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
    elif synthesize_targets_if_empty and rng is not None and base_vertices:
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
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> Tuple[bool, Optional[Set[PlayerId]], Optional[Dict[PlayerId, int]]]:
    """Exact strong-core verification using the joint coalition separator."""
    result = separate_blocking_coalition(
        solution,
        cycle_db,
        partition,
        max_coal_size,
        Delta,
        core_type="strong",
        solver=solver,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        solver_seed=solver_seed,
    )
    in_core = result["blocking"] is False and bool(result["certified"])
    coalition = result["coalition"] if result["blocking"] else None
    return in_core, coalition, result["baseline_utilities"]


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
    donor_order: Optional[Sequence[VertexId]] = None,
) -> Dict[str, object]:
    """
    Heuristic identical to core_heuristic but using strong_core_verification.
    Adds altruists only if the IP with accumulated cuts becomes infeasible.
    """
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}

    existing_ids = set(vertices) | set(adj_out.keys())
    available_altruists = _ordered_altruists(
        altruist_edges, set(existing_ids), donor_order, rng
    )
    full_donor_order = list(available_altruists)

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
                        real_vertices=set(partition.owner_of),
                    )
                else:
                    new_altruist = max(
                        set(working_adj.keys()).union(current_vertices).union(
                            v for nbrs in working_adj.values() for v in nbrs
                        ),
                        default=-1,
                    ) + 1
                    _add_altruist_vertex(
                        working_adj,
                        current_vertices,
                        new_altruist,
                        [],
                        rng=rng,
                        real_vertices=set(partition.owner_of),
                        synthesize_targets_if_empty=True,
                    )

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
    used_altruists = _used_altruists(best_solution, final_cycle_db, altruists)
    return {
        "solution": best_solution,
        "altruists_added": len(altruists),
        "altruists_introduced": len(altruists),
        "altruists_used": len(used_altruists),
        "introduced_donor_ids": list(altruists),
        "used_donor_ids": sorted(used_altruists),
        "donor_order": full_donor_order,
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
    donor_order: Optional[Sequence[VertexId]] = None,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
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
    available_altruists = _ordered_altruists(
        altruist_edges, set(existing_ids), donor_order, rng
    )
    full_donor_order = list(available_altruists)

    # Work on a mutable copy of the adjacency
    working_adj = {u: list(neigh) for u, neigh in adj_out.items()}
    current_vertices = list(vertices)
    altruists: List[VertexId] = []
    cuts: List[Dict[str, object]] = []

    # Initial cycle DB and solve
    cycle_db = enumerate_cycles(current_vertices + altruists, working_adj, partition, Delta, altruists)
    solution, status = _solve_cycle_ip(
        cycle_db,
        Delta,
        partition,
        solver,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        solver_seed=solver_seed,
    )

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
        separation = separate_blocking_coalition(
            solution,
            cycle_db,
            partition,
            max_coal_size,
            Delta,
            core_type="weak",
            solver=solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        if not bool(separation.get("certified")):
            terminated = True
            break
        if separation.get("blocking") is False:
            best_solution = solution
            final_in_core = True
            break

        # Add coalition cut: total utility of S must be at least previous + 1
        blocking_coalition = separation.get("coalition")
        utilities = separation.get("baseline_utilities", {})
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
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                solver_seed=solver_seed,
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
                        real_vertices=set(partition.owner_of),
                    )
                else:
                    # Fallback synthetic altruist id with no preset targets (optional)
                    new_altruist = max(
                        set(working_adj.keys()).union(current_vertices).union(
                            v for nbrs in working_adj.values() for v in nbrs
                        ),
                        default=-1,
                    ) + 1
                    _add_altruist_vertex(
                        working_adj,
                        current_vertices,
                        new_altruist,
                        [],
                        rng=rng,
                        real_vertices=set(partition.owner_of),
                        synthesize_targets_if_empty=True,
                    )

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
    used_altruists = _used_altruists(best_solution, final_cycle_db, altruists)
    return {
        "solution": best_solution,
        "altruists_added": len(altruists),
        "altruists_introduced": len(altruists),
        "altruists_used": len(used_altruists),
        "introduced_donor_ids": list(altruists),
        "used_donor_ids": sorted(used_altruists),
        "donor_order": full_donor_order,
        "cuts_used": cuts,
        "final_in_core": final_in_core,
        "player_utilities": player_utilities,
        "objective_value": objective_value,
    }


# --- TU Core (Simple) with Altruists ----------------------------------------------------------


def _ordered_altruists(
    altruist_edges: Mapping[VertexId, Sequence[VertexId]],
    existing_ids: Set[VertexId],
    donor_order: Optional[Sequence[VertexId]],
    rng: np.random.Generator,
) -> List[VertexId]:
    """Return one validated, reproducible permutation of available altruists."""
    available = [int(a) for a in altruist_edges if int(a) not in existing_ids]
    available_set = set(available)
    if donor_order is None:
        ordered = sorted(available)
        rng.shuffle(ordered)
        return ordered
    ordered = [int(a) for a in donor_order]
    if len(ordered) != len(set(ordered)):
        raise ValueError("donor_order contains duplicate donor ids")
    if set(ordered) != available_set:
        missing = sorted(available_set.difference(ordered))
        unknown = sorted(set(ordered).difference(available_set))
        raise ValueError(
            f"donor_order must be a permutation of available altruists; "
            f"missing={missing}, unknown={unknown}"
        )
    return ordered


def _used_altruists(
    solution: Set[CycleId],
    cycle_db: CycleDB,
    introduced: Iterable[VertexId],
) -> Set[VertexId]:
    introduced_set = set(int(a) for a in introduced)
    return {
        vertex
        for cycle_id in solution
        for vertex in cycle_db.cycles[cycle_id].vertices
        if vertex in introduced_set
    }


def core_tu_simple(
    vertices: List[VertexId],
    adj_out: AdjOut,
    partition: Partition,
    Delta: int,
    max_coal_size: int = 3,
    solver: str = "GUROBI",
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = 0.0,
    rng: Optional[np.random.Generator] = None,
    altruist_edges: Optional[Mapping[VertexId, Sequence[VertexId]]] = None,
    donor_order: Optional[Sequence[VertexId]] = None,
    threads: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Find a capped TU-core allocation with sequential donor introduction.

    The donor-free maximum number of real-patient transplants is used as a
    coverage floor.  Valid aggregate TU coalition cuts are generated by a
    single joint separation MIP.  If those cuts make the master infeasible,
    donors are introduced one at a time in ``donor_order`` and the exact TU
    check is repeated.  The returned donor counts distinguish donors made
    available from donors actually appearing in the selected exchange.
    """
    if Delta not in (2, 3):
        raise ValueError("Delta must be 2 or 3")
    if max_coal_size <= 0:
        raise ValueError("max_coal_size must be positive")

    run_started = perf_counter()
    rng = rng or np.random.default_rng()
    altruist_edges = altruist_edges or {}
    real_vertex_set = set(int(v) for v in vertices)
    existing_ids = real_vertex_set | set(int(v) for v in adj_out)
    ordered_donors = _ordered_altruists(
        altruist_edges,
        existing_ids,
        donor_order,
        rng,
    )

    cycle_enumeration_seconds = 0.0
    optimization_seconds = 0.0
    separation_seconds = 0.0
    master_solves = 0
    separation_solves = 0
    solve_gaps: List[float] = []
    history: List[Dict[str, object]] = []

    enumeration_started = perf_counter()
    base_cycle_db = enumerate_cycles(vertices, adj_out, partition, Delta)
    cycle_enumeration_seconds += perf_counter() - enumeration_started
    base_cycles = [cycle for cycle in base_cycle_db.cycles if cycle.length <= Delta]
    base_problem = pulp.LpProblem("TUCoreBaseline", pulp.LpMaximize)
    base_vars = {
        cycle.id: pulp.LpVariable(f"base_{cycle.id}", lowBound=0, upBound=1, cat="Binary")
        for cycle in base_cycles
    }
    base_problem += pulp.lpSum(
        base_vars[cycle.id] * cycle.non_altruist_count for cycle in base_cycles
    )
    for vertex, cycle_ids in base_cycle_db.by_vertex.items():
        relevant = [base_vars[cid] for cid in cycle_ids if cid in base_vars]
        if relevant:
            base_problem += pulp.lpSum(relevant) <= 1, f"base_disjoint_{vertex}"
    base_solver = make_pulp_solver(
        solver,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        solver_seed=solver_seed,
    )
    base_diag = _solve_with_diagnostics(base_problem, base_solver)
    optimization_seconds += float(base_diag["runtime_seconds"])
    master_solves += 1
    if base_diag.get("mip_gap") is not None:
        solve_gaps.append(float(base_diag["mip_gap"]))
    if base_problem.status != pulp.LpStatusOptimal:
        return {
            "solution": set(),
            "in_core": False,
            "certified": False,
            "termination_reason": "baseline_not_optimal",
            "solver_status": base_diag["status"],
            "player_utilities": {},
            "objective_value": 0,
            "objective_real_patients": 0,
            "baseline_real_transplants": None,
            "altruists_present": False,
            "altruists_introduced": 0,
            "altruists_added": 0,
            "altruists_used": 0,
            "introduced_donor_ids": [],
            "used_donor_ids": [],
            "donor_order": ordered_donors,
            "num_coalitions": 0,
            "num_cuts": 0,
            "num_master_solves": master_solves,
            "num_separation_solves": 0,
            "runtime_seconds": perf_counter() - run_started,
            "cycle_enumeration_seconds": cycle_enumeration_seconds,
            "optimization_seconds": optimization_seconds,
            "separation_seconds": separation_seconds,
            "max_mip_gap": max(solve_gaps) if solve_gaps else None,
            "history": history,
        }

    base_solution = {
        cycle_id
        for cycle_id, variable in base_vars.items()
        if variable.value() is not None and variable.value() > 0.5
    }
    baseline_real_transplants = sum(
        base_cycle_db.cycles[cycle_id].non_altruist_count
        for cycle_id in base_solution
    )

    working_adj = {u: list(neighbors) for u, neighbors in adj_out.items()}
    current_vertices = list(vertices)
    introduced: List[VertexId] = []
    remaining_donors = list(ordered_donors)
    cuts: List[Dict[str, object]] = []
    cut_keys: Set[Tuple[Tuple[PlayerId, ...], int]] = set()

    def finish(
        *,
        solution: Set[CycleId],
        cycle_db: CycleDB,
        in_core: bool,
        certified: bool,
        reason: str,
        solver_status: str,
    ) -> Dict[str, object]:
        used = _used_altruists(solution, cycle_db, introduced)
        real_transplants = sum(
            cycle_db.cycles[cycle_id].non_altruist_count for cycle_id in solution
        )
        return {
            "solution": solution,
            "in_core": bool(in_core),
            "certified": bool(certified),
            "termination_reason": reason,
            "solver_status": solver_status,
            "player_utilities": compute_player_utilities(solution, cycle_db),
            "objective_value": len(used),
            "objective_real_patients": int(real_transplants),
            "objective_altruist_penalty": len(used),
            "baseline_real_transplants": int(baseline_real_transplants),
            "num_coalitions": len(cuts),
            "num_cuts": len(cuts),
            "max_coal_size": max_coal_size,
            "altruists_present": bool(introduced),
            "M": len(set(current_vertices)),
            "altruists_introduced": len(introduced),
            "altruists_added": len(introduced),
            "altruists_used": len(used),
            "introduced_donor_ids": list(introduced),
            "used_donor_ids": sorted(used),
            "donor_order": list(ordered_donors),
            "num_master_solves": master_solves,
            "num_separation_solves": separation_solves,
            "runtime_seconds": perf_counter() - run_started,
            "cycle_enumeration_seconds": cycle_enumeration_seconds,
            "optimization_seconds": optimization_seconds,
            "separation_seconds": separation_seconds,
            "max_mip_gap": max(solve_gaps) if solve_gaps else 0.0,
            "history": history,
        }

    while True:
        enumeration_started = perf_counter()
        cycle_db = enumerate_cycles(
            current_vertices,
            working_adj,
            partition,
            Delta,
            introduced,
        )
        cycle_enumeration_seconds += perf_counter() - enumeration_started
        cycles = [cycle for cycle in cycle_db.cycles if cycle.length <= Delta]
        problem = pulp.LpProblem("TUCoreCuttingPlane", pulp.LpMinimize)
        x_vars = {
            cycle.id: pulp.LpVariable(f"master_{cycle.id}", lowBound=0, upBound=1, cat="Binary")
            for cycle in cycles
        }
        problem += pulp.lpSum(
            x_vars[cycle.id] * cycle.altruist_count for cycle in cycles
        )
        for vertex, cycle_ids in cycle_db.by_vertex.items():
            relevant = [x_vars[cid] for cid in cycle_ids if cid in x_vars]
            if relevant:
                problem += pulp.lpSum(relevant) <= 1, f"master_disjoint_{vertex}"
        problem += (
            pulp.lpSum(
                x_vars[cycle.id] * cycle.non_altruist_count for cycle in cycles
            )
            >= baseline_real_transplants
        ), "coverage_floor"
        for cut_index, cut in enumerate(cuts):
            coalition = set(cut["coalition"])
            lhs = [
                x_vars[cycle.id]
                * sum(cycle.player_counts.get(player, 0) for player in coalition)
                for cycle in cycles
                if any(cycle.player_counts.get(player, 0) for player in coalition)
            ]
            problem += pulp.lpSum(lhs) >= int(cut["rhs"]), f"tu_cut_{cut_index}"

        master_solver = make_pulp_solver(
            solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        master_diag = _solve_with_diagnostics(problem, master_solver)
        optimization_seconds += float(master_diag["runtime_seconds"])
        master_solves += 1
        if master_diag.get("mip_gap") is not None:
            solve_gaps.append(float(master_diag["mip_gap"]))
        selected = {
            cycle_id
            for cycle_id, variable in x_vars.items()
            if variable.value() is not None and variable.value() > 0.5
        }
        history.append(
            {
                "donors_introduced": len(introduced),
                "phase": "master",
                "status": master_diag["status"],
                "runtime_seconds": master_diag["runtime_seconds"],
                "num_cuts": len(cuts),
            }
        )

        if problem.status == pulp.LpStatusInfeasible:
            if not remaining_donors:
                return finish(
                    solution=set(),
                    cycle_db=cycle_db,
                    in_core=False,
                    certified=True,
                    reason="infeasible_after_all_donors",
                    solver_status=str(master_diag["status"]),
                )
            new_altruist = remaining_donors.pop(0)
            targets = [
                int(target)
                for target in altruist_edges.get(new_altruist, [])
                if int(target) in real_vertex_set
            ]
            _add_altruist_vertex(
                working_adj,
                current_vertices,
                new_altruist,
                targets,
                rng=rng,
                real_vertices=real_vertex_set,
            )
            introduced.append(new_altruist)
            current_vertices.append(new_altruist)
            continue

        if problem.status != pulp.LpStatusOptimal:
            return finish(
                solution=selected,
                cycle_db=cycle_db,
                in_core=False,
                certified=False,
                reason="master_not_optimal",
                solver_status=str(master_diag["status"]),
            )

        separation_started = perf_counter()
        separation = separate_blocking_coalition(
            selected,
            cycle_db,
            partition,
            max_coal_size,
            Delta,
            core_type="tu",
            solver=solver,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_seed=solver_seed,
        )
        separation_seconds += perf_counter() - separation_started
        separation_solves += 1
        separation_diag = separation["diagnostics"]
        if separation_diag.get("mip_gap") is not None:
            solve_gaps.append(float(separation_diag["mip_gap"]))
        history.append(
            {
                "donors_introduced": len(introduced),
                "phase": "separation",
                "status": separation_diag["status"],
                "runtime_seconds": separation_diag["runtime_seconds"],
                "blocking": separation["blocking"],
                "coalition_size": len(separation["coalition"]),
                "violation": separation["violation"],
            }
        )

        if separation["blocking"] is True:
            coalition_tuple = tuple(sorted(separation["coalition"]))
            rhs = int(separation["deviation_value"])
            cut_key = (coalition_tuple, rhs)
            if cut_key in cut_keys:
                return finish(
                    solution=selected,
                    cycle_db=cycle_db,
                    in_core=False,
                    certified=False,
                    reason="duplicate_violated_cut",
                    solver_status=str(separation_diag["status"]),
                )
            cut_keys.add(cut_key)
            cuts.append({"coalition": coalition_tuple, "rhs": rhs})
            continue

        if separation["blocking"] is False and separation["certified"]:
            return finish(
                solution=selected,
                cycle_db=cycle_db,
                in_core=True,
                certified=True,
                reason="stable",
                solver_status=str(master_diag["status"]),
            )

        return finish(
            solution=selected,
            cycle_db=cycle_db,
            in_core=False,
            certified=False,
            reason="separation_not_certified",
            solver_status=str(separation_diag["status"]),
        )
