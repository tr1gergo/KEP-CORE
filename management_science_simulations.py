"""Reproducible, resumable driver for the Management Science simulation study.

The driver deliberately separates three sources of randomness:

* the compatibility-graph subsample depends only on base pool and pool size;
* the organizational partition also depends on organization count/replication;
* donor permutations depend on the market, but not on the algorithm or cycle cap.

Raw results are appended after every algorithm/order run, so an interrupted
study can resume without repeating completed work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import platform
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pulp

from instance_analysis import load_instance
from KEP_functions import (
    GraphFeatures,
    Partition,
    build_compat_graph,
    build_graph_features,
    core_heuristic,
    core_tu_simple,
    enumerate_cycles,
    lexicographic_core_search,
    make_partition,
    separate_blocking_coalition,
    strong_core_heuristic,
)


@dataclass(frozen=True)
class StudyConfig:
    instance_dir: str = "instances_large"
    output_dir: str = "results/management_science_revised"
    num_base_instances: int = 30
    instance_selection_seed: int = 20260819
    master_seed: int = 20260819
    pool_sizes: Tuple[int, ...] = (100, 200, 500)
    num_players: Tuple[int, ...] = (5, 10, 15, 20, 30)
    deltas: Tuple[int, ...] = (2, 3)
    partition_reps: int = 2
    partition_var_size: int = 1
    donor_order_reps: int = 3
    max_coal_size: int = 4
    solver: str = "GUROBI"
    time_limit_seconds: Optional[int] = 300
    mip_gap: float = 0.0
    solver_threads: int = 1
    run_heuristic_diagnostics: bool = True
    run_legacy_lexicographic_donor_search: bool = False
    max_cells: Optional[int] = None


def stable_seed(*items: object) -> int:
    """Stable 32-bit seed; unlike ``hash()``, it is invariant across processes."""
    digest = sha256("|".join(map(str, items)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _instance_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 10**12


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_base_instances(config: StudyConfig) -> List[Path]:
    """Select exactly the preregistered number of unique base pools."""
    instance_dir = Path(config.instance_dir)
    candidates = sorted(instance_dir.glob("genxml-*.xml"), key=_instance_number)
    if len(candidates) < config.num_base_instances:
        raise ValueError(
            f"Requested {config.num_base_instances} base pools but found {len(candidates)}"
        )
    rng = np.random.default_rng(config.instance_selection_seed)
    selected_indices = sorted(
        int(index)
        for index in rng.choice(
            len(candidates), size=config.num_base_instances, replace=False
        )
    )
    return [candidates[index] for index in selected_indices]


def write_study_manifest(config: StudyConfig, paths: Sequence[Path]) -> Path:
    """Freeze configuration, file identities, and computing environment."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "study_manifest.json"
    normalized_config = json.loads(json.dumps(asdict(config)))
    code_paths = [
        Path("KEP_functions.py"),
        Path("management_science_simulations.py"),
    ]
    payload = {
        "config": normalized_config,
        "instances": [
            {
                "path": path.as_posix(),
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in paths
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pulp": pulp.__version__,
        },
        "code": {
            path.as_posix(): _sha256_file(path)
            for path in code_paths
            if path.exists()
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("config") != payload["config"]
            or existing.get("instances") != payload["instances"]
            or existing.get("code") != payload["code"]
        ):
            raise ValueError(
                f"Existing manifest {manifest_path} does not match this design. "
                "Use a new output_dir for a different study."
            )
    else:
        manifest_path.write_text(serialized + "\n", encoding="utf-8")
    return manifest_path


def make_donor_orders(
    altruist_edges: Mapping[int, Sequence[int]],
    *,
    master_seed: int,
    instance_name: str,
    pool_size: int,
    num_players: int,
    partition_rep: int,
    repetitions: int,
) -> List[List[int]]:
    """Create common donor permutations reused across algorithms and deltas."""
    donor_ids = np.array(sorted(int(donor) for donor in altruist_edges), dtype=int)
    orders: List[List[int]] = []
    for order_rep in range(repetitions):
        rng = np.random.default_rng(
            stable_seed(
                master_seed,
                "donor_order",
                instance_name,
                pool_size,
                num_players,
                partition_rep,
                order_rep,
            )
        )
        orders.append([int(donor) for donor in rng.permutation(donor_ids)])
    return orders


def _result_key(row: Mapping[str, object]) -> Tuple[str, str, int]:
    return (
        str(row["market_id"]),
        str(row["algorithm"]),
        int(row.get("order_rep", 0)),
    )


def _load_checkpoint(path: Path) -> Dict[Tuple[str, str, int], Dict[str, object]]:
    rows: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid checkpoint line {line_number}: {exc}") from exc
            rows[_result_key(row)] = row
    return rows


def _append_checkpoint(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _market_base(
    *,
    instance_name: str,
    pool_size: int,
    num_players: int,
    partition_rep: int,
    delta: int,
    actual_vertices: int,
    available_altruists: int,
    graph_seed: int,
    partition_seed: int,
    graph_seconds: float,
    partition_seconds: float,
) -> Dict[str, object]:
    market_id = (
        f"{instance_name}|p{pool_size}|n{num_players}|"
        f"r{partition_rep}|d{delta}"
    )
    return {
        "market_id": market_id,
        "instance": instance_name,
        "pool_size": int(pool_size),
        "num_players": int(num_players),
        "partition_rep": int(partition_rep),
        "Delta": int(delta),
        "actual_vertices": int(actual_vertices),
        "available_altruists": int(available_altruists),
        "graph_seed": int(graph_seed),
        "partition_seed": int(partition_seed),
        "graph_construction_seconds": float(graph_seconds),
        "partition_seconds": float(partition_seconds),
    }


def _flatten_result(
    base: Mapping[str, object],
    *,
    algorithm: str,
    order_rep: int,
    evidence_type: str,
    result: Mapping[str, object],
    runner_seconds: float,
) -> Dict[str, object]:
    in_core = result.get("in_core", result.get("final_in_core"))
    introduced = result.get(
        "altruists_introduced", result.get("altruists_added", 0)
    )
    used = result.get("altruists_used")
    row = dict(base)
    row.update(
        {
            "algorithm": algorithm,
            "order_rep": int(order_rep),
            "evidence_type": evidence_type,
            "in_core": None if in_core is None else bool(in_core),
            "certified": bool(result.get("certified", False)),
            "termination_reason": result.get("termination_reason"),
            "solver_status": result.get("solver_status"),
            "donors_introduced": None if introduced is None else int(introduced),
            "donors_used": None if used is None else int(used),
            "introduced_donor_ids": list(result.get("introduced_donor_ids", [])),
            "used_donor_ids": list(result.get("used_donor_ids", [])),
            "donor_order": list(result.get("donor_order", [])),
            "real_transplants": result.get(
                "objective_real_patients",
                result.get("objective_tiers", {}).get("transplants"),
            ),
            "baseline_real_transplants": result.get("baseline_real_transplants"),
            "runtime_seconds": float(result.get("runtime_seconds", runner_seconds)),
            "runner_seconds": float(runner_seconds),
            "cycle_enumeration_seconds": result.get("cycle_enumeration_seconds"),
            "optimization_seconds": result.get("optimization_seconds"),
            "separation_seconds": result.get("separation_seconds"),
            "num_master_solves": result.get("num_master_solves"),
            "num_optimization_solves": result.get("num_optimization_solves"),
            "num_separation_solves": result.get("num_separation_solves"),
            "num_cuts": result.get("num_cuts", len(result.get("cuts_used", []))),
            "max_mip_gap": result.get("max_mip_gap"),
        }
    )
    return row


def _inferred_core_row(
    base: Mapping[str, object],
    *,
    algorithm: str,
    order_rep: int,
    tu_row: Mapping[str, object],
) -> Dict[str, object]:
    row = dict(base)
    introduced = int(tu_row.get("donors_introduced") or 0)
    evidence = (
        "inferred_exact_zero_from_tu"
        if introduced == 0
        else "existence_upper_bound_from_tu"
    )
    row.update(
        {
            "algorithm": algorithm,
            "order_rep": int(order_rep),
            "evidence_type": evidence,
            "in_core": True,
            "certified": True,
            "termination_reason": "inferred_from_tu_core_allocation",
            "solver_status": "NotRun",
            "donors_introduced": introduced,
            "donors_used": tu_row.get("donors_used"),
            "introduced_donor_ids": list(tu_row.get("introduced_donor_ids", [])),
            "used_donor_ids": list(tu_row.get("used_donor_ids", [])),
            "donor_order": list(tu_row.get("donor_order", [])),
            "real_transplants": tu_row.get("real_transplants"),
            "baseline_real_transplants": tu_row.get("baseline_real_transplants"),
            "runtime_seconds": 0.0,
            "runner_seconds": 0.0,
            "cycle_enumeration_seconds": 0.0,
            "optimization_seconds": 0.0,
            "separation_seconds": 0.0,
            "num_master_solves": 0,
            "num_optimization_solves": 0,
            "num_separation_solves": 0,
            "num_cuts": 0,
            "max_mip_gap": 0.0,
        }
    )
    return row


def _lex_cap_sensitivity(
    result: Mapping[str, object],
    vertices: List[int],
    adj_out: Dict[int, List[int]],
    partition: Partition,
    delta: int,
    config: StudyConfig,
    solver_seed: int,
) -> Dict[str, object]:
    """Infer or solve k=2,3,4 stability for the donor-free lex allocation."""
    if not result.get("certified"):
        return {
            "blocked_k2": None,
            "blocked_k3": None,
            "blocked_k4": None,
            "cap_sensitivity_certified": False,
            "cap_sensitivity_seconds": 0.0,
        }
    if result.get("final_in_core"):
        return {
            "blocked_k2": False,
            "blocked_k3": False,
            "blocked_k4": False,
            "cap_sensitivity_certified": True,
            "cap_sensitivity_seconds": 0.0,
        }

    cycle_db = enumerate_cycles(vertices, adj_out, partition, delta)
    solution = set(result.get("solution", set()))
    started = perf_counter()
    cap_results: Dict[int, Optional[bool]] = {}
    for cap in (2, 3, 4):
        if any(value is True for value in cap_results.values()):
            cap_results[cap] = True
            continue
        if cap == config.max_coal_size:
            # The allocation was already exactly shown to be unstable at the
            # configured cap by ``lexicographic_core_search``.
            cap_results[cap] = True
            continue
        if cap > config.max_coal_size:
            cap_results[cap] = None
            continue
        separation = separate_blocking_coalition(
            solution,
            cycle_db,
            partition,
            cap,
            delta,
            core_type="weak",
            solver=config.solver,
            time_limit=config.time_limit_seconds,
            mip_gap=config.mip_gap,
            threads=config.solver_threads,
            solver_seed=solver_seed,
        )
        if separation["blocking"] is None:
            cap_results[cap] = None
        else:
            cap_results[cap] = bool(separation["blocking"])
    certified = all(value is not None for value in cap_results.values())
    return {
        "blocked_k2": cap_results.get(2),
        "blocked_k3": cap_results.get(3),
        "blocked_k4": cap_results.get(4),
        "cap_sensitivity_certified": certified,
        "cap_sensitivity_seconds": perf_counter() - started,
    }


def _run_call(function, /, *args, **kwargs) -> Tuple[Dict[str, object], float]:
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started


def run_study(config: StudyConfig = StudyConfig()) -> Dict[str, Path]:
    """Run or resume the preregistered experiment without recomputing rows."""
    selected_paths = select_base_instances(config)
    manifest_path = write_study_manifest(config, selected_paths)
    output_dir = Path(config.output_dir)
    raw_path = output_dir / "raw_results.jsonl"
    csv_path = output_dir / "raw_results.csv"
    completed = _load_checkpoint(raw_path)

    total_cells = (
        len(selected_paths)
        * len(config.pool_sizes)
        * len(config.num_players)
        * config.partition_reps
        * len(config.deltas)
    )
    cell_limit = min(total_cells, config.max_cells) if config.max_cells else total_cells
    cells_seen = 0
    study_started = perf_counter()

    def save(row: Dict[str, object]) -> Dict[str, object]:
        key = _result_key(row)
        if key not in completed:
            _append_checkpoint(raw_path, row)
            completed[key] = row
        return completed[key]

    for instance_path in selected_paths:
        load_started = perf_counter()
        instance = load_instance(instance_path)
        instance_load_seconds = perf_counter() - load_started
        features: GraphFeatures = build_graph_features(instance)

        for pool_size in config.pool_sizes:
            graph_seed = stable_seed(
                config.master_seed, "graph", instance_path.name, pool_size
            )
            graph_started = perf_counter()
            vertices, adj_out, _, altruist_edges = build_compat_graph(
                instance,
                num_patients=pool_size,
                rng=np.random.default_rng(graph_seed),
            )
            graph_seconds = perf_counter() - graph_started

            for num_players in config.num_players:
                for partition_rep in range(config.partition_reps):
                    partition_seed = stable_seed(
                        config.master_seed,
                        "partition",
                        instance_path.name,
                        pool_size,
                        num_players,
                        partition_rep,
                    )
                    partition_started = perf_counter()
                    partition = make_partition(
                        vertices,
                        num_players=num_players,
                        var_size=config.partition_var_size,
                        rng=np.random.default_rng(partition_seed),
                    )
                    partition_seconds = perf_counter() - partition_started
                    donor_orders = make_donor_orders(
                        altruist_edges,
                        master_seed=config.master_seed,
                        instance_name=instance_path.name,
                        pool_size=pool_size,
                        num_players=num_players,
                        partition_rep=partition_rep,
                        repetitions=config.donor_order_reps,
                    )

                    for delta in config.deltas:
                        if cells_seen >= cell_limit:
                            frame = pd.DataFrame(completed.values())
                            frame.to_csv(csv_path, index=False)
                            return {
                                "manifest": manifest_path,
                                "raw_jsonl": raw_path,
                                "raw_csv": csv_path,
                            }
                        cells_seen += 1
                        solver_seed = stable_seed(
                            config.master_seed,
                            "solver",
                            instance_path.name,
                            pool_size,
                            num_players,
                            partition_rep,
                            delta,
                        )
                        base = _market_base(
                            instance_name=instance_path.name,
                            pool_size=pool_size,
                            num_players=num_players,
                            partition_rep=partition_rep,
                            delta=delta,
                            actual_vertices=len(vertices),
                            available_altruists=len(altruist_edges),
                            graph_seed=graph_seed,
                            partition_seed=partition_seed,
                            graph_seconds=graph_seconds,
                            partition_seconds=partition_seconds,
                        )
                        base["instance_load_seconds"] = instance_load_seconds
                        market_id = str(base["market_id"])

                        # TU: run one order first; only branch to the other two
                        # if the donor-free capped TU core was not found.
                        tu_rows: List[Dict[str, object]] = []
                        first_key = (market_id, "tu", 0)
                        if first_key in completed:
                            first_tu = completed[first_key]
                        else:
                            result, elapsed = _run_call(
                                core_tu_simple,
                                vertices,
                                adj_out,
                                partition,
                                delta,
                                max_coal_size=config.max_coal_size,
                                solver=config.solver,
                                time_limit=config.time_limit_seconds,
                                mip_gap=config.mip_gap,
                                rng=np.random.default_rng(
                                    stable_seed(solver_seed, "tu", 0)
                                ),
                                altruist_edges=altruist_edges,
                                donor_order=donor_orders[0],
                                threads=config.solver_threads,
                                solver_seed=stable_seed(solver_seed, "tu_solver", 0),
                            )
                            first_tu = save(
                                _flatten_result(
                                    base,
                                    algorithm="tu",
                                    order_rep=0,
                                    evidence_type="exact_capped_tu",
                                    result=result,
                                    runner_seconds=elapsed,
                                )
                            )
                        tu_rows.append(first_tu)
                        tu_needs_order_study = bool(first_tu.get("certified")) and not (
                            bool(first_tu.get("in_core"))
                            and int(first_tu.get("donors_introduced") or 0) == 0
                        )
                        if tu_needs_order_study:
                            for order_rep in range(1, config.donor_order_reps):
                                key = (market_id, "tu", order_rep)
                                if key in completed:
                                    row = completed[key]
                                else:
                                    result, elapsed = _run_call(
                                        core_tu_simple,
                                        vertices,
                                        adj_out,
                                        partition,
                                        delta,
                                        max_coal_size=config.max_coal_size,
                                        solver=config.solver,
                                        time_limit=config.time_limit_seconds,
                                        mip_gap=config.mip_gap,
                                        rng=np.random.default_rng(
                                            stable_seed(solver_seed, "tu", order_rep)
                                        ),
                                        altruist_edges=altruist_edges,
                                        donor_order=donor_orders[order_rep],
                                        threads=config.solver_threads,
                                        solver_seed=stable_seed(
                                            solver_seed, "tu_solver", order_rep
                                        ),
                                    )
                                    row = save(
                                        _flatten_result(
                                            base,
                                            algorithm="tu",
                                            order_rep=order_rep,
                                            evidence_type="exact_capped_tu",
                                            result=result,
                                            runner_seconds=elapsed,
                                        )
                                    )
                                tu_rows.append(row)

                        # A donor-free TU-core allocation is automatically in the
                        # donor-free strong and weak cores.  This exact zero-donor
                        # implication avoids both redundant searches.  If TU uses
                        # donors, the other cores may still need fewer, so their
                        # diagnostic searches remain informative.
                        donor_free_tu_rows = [
                            row
                            for row in tu_rows
                            if row.get("certified")
                            and row.get("in_core")
                            and int(row.get("donors_introduced") or 0) == 0
                        ]
                        for tu_row in donor_free_tu_rows:
                            order_rep = int(tu_row["order_rep"])
                            for algorithm in ("strong", "weak"):
                                save(
                                    _inferred_core_row(
                                        base,
                                        algorithm=algorithm,
                                        order_rep=order_rep,
                                        tu_row=tu_row,
                                    )
                                )

                        # Only when donor-free TU stability was not established do
                        # the aggregate-cut strong/weak heuristics add information.
                        if config.run_heuristic_diagnostics and not donor_free_tu_rows:
                            strong_rows: List[Dict[str, object]] = []
                            for order_rep in range(config.donor_order_reps):
                                if order_rep > 0 and strong_rows and int(
                                    strong_rows[0].get("donors_introduced") or 0
                                ) == 0:
                                    break
                                key = (market_id, "strong_heuristic", order_rep)
                                if key in completed:
                                    row = completed[key]
                                else:
                                    result, elapsed = _run_call(
                                        strong_core_heuristic,
                                        vertices,
                                        adj_out,
                                        partition,
                                        delta,
                                        solver=config.solver,
                                        max_coal_size=config.max_coal_size,
                                        max_altruists=len(altruist_edges),
                                        rng=np.random.default_rng(
                                            stable_seed(solver_seed, "strong", order_rep)
                                        ),
                                        altruist_edges=altruist_edges,
                                        donor_order=donor_orders[order_rep],
                                    )
                                    result = dict(result)
                                    result["certified"] = bool(
                                        result.get("final_in_core")
                                    )
                                    result["termination_reason"] = (
                                        "heuristic_found_stable_allocation"
                                        if result.get("final_in_core")
                                        else "heuristic_did_not_find_stable_allocation"
                                    )
                                    row = save(
                                        _flatten_result(
                                            base,
                                            algorithm="strong_heuristic",
                                            order_rep=order_rep,
                                            evidence_type="aggregate_cut_heuristic",
                                            result=result,
                                            runner_seconds=elapsed,
                                        )
                                    )
                                strong_rows.append(row)
                            donor_free_strong_rows = [
                                row
                                for row in strong_rows
                                if row.get("in_core")
                                and int(row.get("donors_introduced") or 0) == 0
                            ]
                            for strong_row in donor_free_strong_rows:
                                if strong_row.get("in_core"):
                                    inferred = dict(strong_row)
                                    inferred["algorithm"] = "weak"
                                    inferred["evidence_type"] = "inferred_from_strong_allocation"
                                    inferred["runtime_seconds"] = 0.0
                                    inferred["runner_seconds"] = 0.0
                                    save(inferred)

                            if not donor_free_strong_rows:
                                weak_rows: List[Dict[str, object]] = []
                                for order_rep in range(config.donor_order_reps):
                                    if order_rep > 0 and weak_rows and int(
                                        weak_rows[0].get("donors_introduced") or 0
                                    ) == 0:
                                        break
                                    key = (market_id, "weak_heuristic", order_rep)
                                    if key in completed:
                                        row = completed[key]
                                    else:
                                        result, elapsed = _run_call(
                                            core_heuristic,
                                            vertices,
                                            adj_out,
                                            partition,
                                            delta,
                                            solver=config.solver,
                                            max_coal_size=config.max_coal_size,
                                            max_altruists=len(altruist_edges),
                                            rng=np.random.default_rng(
                                                stable_seed(solver_seed, "weak", order_rep)
                                            ),
                                            altruist_edges=altruist_edges,
                                            donor_order=donor_orders[order_rep],
                                        )
                                        result = dict(result)
                                        result["certified"] = bool(
                                            result.get("final_in_core")
                                        )
                                        result["termination_reason"] = (
                                            "heuristic_found_stable_allocation"
                                            if result.get("final_in_core")
                                            else "heuristic_did_not_find_stable_allocation"
                                        )
                                        row = save(
                                            _flatten_result(
                                                base,
                                                algorithm="weak_heuristic",
                                                order_rep=order_rep,
                                                evidence_type="aggregate_cut_heuristic",
                                                result=result,
                                                runner_seconds=elapsed,
                                            )
                                        )
                                    weak_rows.append(row)

                        # Lexicographic rule: first solve with no donor additions.
                        # If stable, this one solve is the final result; otherwise
                        # run the three common donor permutations.
                        lex_key = (market_id, "lexicographic", 0)
                        initial_key = (market_id, "lexicographic_initial", -1)
                        if lex_key in completed and int(
                            completed[lex_key].get("donors_introduced") or 0
                        ) == 0:
                            initial_lex_row = completed[lex_key]
                            initial_unstable = False
                        elif initial_key in completed:
                            initial_lex_row = completed[initial_key]
                            initial_unstable = bool(
                                initial_lex_row.get("certified")
                                and not initial_lex_row.get("in_core")
                            )
                        else:
                            initial_result, elapsed = _run_call(
                                lexicographic_core_search,
                                vertices,
                                adj_out,
                                partition,
                                delta,
                                graph_features=features,
                                max_coal_size=config.max_coal_size,
                                solver=config.solver,
                                altruist_edges=altruist_edges,
                                max_added_altruists=0,
                                rng=np.random.default_rng(
                                    stable_seed(solver_seed, "lex_initial")
                                ),
                                donor_order=donor_orders[0],
                                time_limit=config.time_limit_seconds,
                                mip_gap=config.mip_gap,
                                threads=config.solver_threads,
                                solver_seed=stable_seed(
                                    solver_seed, "lex_initial_solver"
                                ),
                            )
                            cap_fields = _lex_cap_sensitivity(
                                initial_result,
                                vertices,
                                adj_out,
                                partition,
                                delta,
                                config,
                                solver_seed,
                            )
                            initial_unstable = bool(
                                initial_result.get("certified")
                                and not initial_result.get("final_in_core")
                            )
                            initial_algorithm = (
                                "lexicographic_initial"
                                if initial_unstable
                                else "lexicographic"
                            )
                            initial_order_rep = -1 if initial_unstable else 0
                            initial_lex_row = _flatten_result(
                                base,
                                algorithm=initial_algorithm,
                                order_rep=initial_order_rep,
                                evidence_type="exact_selected_allocation_check",
                                result=initial_result,
                                runner_seconds=elapsed,
                            )
                            initial_lex_row.update(cap_fields)
                            initial_lex_row = save(initial_lex_row)

                        # The final protocol follows a blocked donor-free
                        # lexicographic allocation with the floor-preserving
                        # stabilization procedure.  Re-optimizing all four
                        # tiers after every donor is retained only as an
                        # explicit legacy option and is not part of the final
                        # experiment.
                        if (
                            initial_unstable
                            and config.run_legacy_lexicographic_donor_search
                        ):
                            for order_rep in range(config.donor_order_reps):
                                key = (market_id, "lexicographic", order_rep)
                                if key in completed:
                                    continue
                                result, elapsed = _run_call(
                                    lexicographic_core_search,
                                    vertices,
                                    adj_out,
                                    partition,
                                    delta,
                                    graph_features=features,
                                    max_coal_size=config.max_coal_size,
                                    solver=config.solver,
                                    altruist_edges=altruist_edges,
                                    rng=np.random.default_rng(
                                        stable_seed(solver_seed, "lex", order_rep)
                                    ),
                                    donor_order=donor_orders[order_rep],
                                    time_limit=config.time_limit_seconds,
                                    mip_gap=config.mip_gap,
                                    threads=config.solver_threads,
                                    solver_seed=stable_seed(
                                        solver_seed, "lex_solver", order_rep
                                    ),
                                )
                                save(
                                    _flatten_result(
                                        base,
                                        algorithm="lexicographic",
                                        order_rep=order_rep,
                                        evidence_type="exact_selected_allocation_check",
                                        result=result,
                                        runner_seconds=elapsed,
                                    )
                                )

                        elapsed_study = perf_counter() - study_started
                        rate = cells_seen / elapsed_study if elapsed_study else 0.0
                        remaining = (cell_limit - cells_seen) / rate if rate else float("inf")
                        print(
                            f"[{cells_seen}/{cell_limit}] {market_id} | "
                            f"elapsed={elapsed_study / 60:.1f} min | "
                            f"ETA={remaining / 60:.1f} min"
                        )

    frame = pd.DataFrame(completed.values())
    frame.to_csv(csv_path, index=False)
    return {
        "manifest": manifest_path,
        "raw_jsonl": raw_path,
        "raw_csv": csv_path,
    }


def load_results(config: StudyConfig = StudyConfig()) -> pd.DataFrame:
    path = Path(config.output_dir) / "raw_results.jsonl"
    return pd.DataFrame(_load_checkpoint(path).values())


def runtime_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Median/tail runtime table including failures and uncertified runs."""
    frame = results.copy()
    frame["runtime_seconds"] = pd.to_numeric(frame["runtime_seconds"], errors="coerce")
    return (
        frame.groupby("algorithm", dropna=False)
        .agg(
            runs=("market_id", "size"),
            certified_rate=("certified", "mean"),
            median_seconds=("runtime_seconds", "median"),
            p90_seconds=("runtime_seconds", lambda values: values.quantile(0.90)),
            p95_seconds=("runtime_seconds", lambda values: values.quantile(0.95)),
            max_seconds=("runtime_seconds", "max"),
        )
        .reset_index()
    )


def donor_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Report unconditional and instability-conditional donor outcomes."""
    frame = results[
        results["algorithm"].isin(["tu", "lexicographic"])
    ].copy()
    frame["donors_introduced"] = pd.to_numeric(
        frame["donors_introduced"], errors="coerce"
    )
    frame["donors_used"] = pd.to_numeric(frame["donors_used"], errors="coerce")

    rows: List[Dict[str, object]] = []
    group_cols = ["algorithm", "pool_size", "num_players", "Delta"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        positive = group[group["donors_introduced"] > 0]
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "runs": len(group),
                "share_introducing_donors": float(
                    (group["donors_introduced"] > 0).mean()
                ),
                "mean_introduced": group["donors_introduced"].mean(),
                "mean_used": group["donors_used"].mean(),
                "conditional_mean_introduced": positive["donors_introduced"].mean(),
                "conditional_median_introduced": positive["donors_introduced"].median(),
                "conditional_min_introduced": positive["donors_introduced"].min(),
                "conditional_max_introduced": positive["donors_introduced"].max(),
                "conditional_mean_used": positive["donors_used"].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def initial_stability_sample(results: pd.DataFrame) -> pd.DataFrame:
    """One donor-free TU and lexicographic stability observation per market."""
    if results.empty:
        return pd.DataFrame()
    tu = results[
        (results["algorithm"] == "tu") & (results["order_rep"] == 0)
    ].copy()
    tu["reported_algorithm"] = "tu"
    tu["donor_free_stable"] = (
        tu["certified"].fillna(False)
        & tu["in_core"].fillna(False)
        & (pd.to_numeric(tu["donors_introduced"], errors="coerce") == 0)
    )

    lex_initial = results[results["algorithm"] == "lexicographic_initial"].copy()
    challenged_market_ids = set(lex_initial["market_id"])
    lex_stable = results[
        (results["algorithm"] == "lexicographic")
        & (results["order_rep"] == 0)
        & (pd.to_numeric(results["donors_introduced"], errors="coerce") == 0)
        & (~results["market_id"].isin(challenged_market_ids))
    ].copy()
    lex = pd.concat([lex_initial, lex_stable], ignore_index=True)
    lex["reported_algorithm"] = "lexicographic"
    lex["donor_free_stable"] = (
        lex["certified"].fillna(False) & lex["in_core"].fillna(False)
    )
    columns = list(
        dict.fromkeys(
            list(tu.columns)
            + ["reported_algorithm", "donor_free_stable"]
        )
    )
    return pd.concat([tu, lex], ignore_index=True).reindex(columns=columns)


def clustered_stability_summary(
    results: pd.DataFrame,
    *,
    bootstrap_reps: int = 2000,
    seed: int = 20260819,
) -> pd.DataFrame:
    """Base-pool clustered 95% intervals for donor-free stability rates."""
    sample = initial_stability_sample(results)
    if sample.empty:
        return sample
    rng = np.random.default_rng(seed)
    group_cols = ["reported_algorithm", "pool_size", "num_players", "Delta"]
    rows: List[Dict[str, object]] = []
    for keys, group in sample.groupby(group_cols, dropna=False):
        certified = group[group["certified"].fillna(False)].copy()
        cluster_means = certified.groupby("instance")["donor_free_stable"].mean()
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        if cluster_means.empty:
            estimate = lower = upper = float("nan")
        else:
            values = cluster_means.to_numpy(dtype=float)
            estimate = float(values.mean())
            draws = rng.choice(
                values,
                size=(int(bootstrap_reps), len(values)),
                replace=True,
            ).mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
        row.update(
            {
                "base_pool_clusters": int(cluster_means.size),
                "certified_partition_rows": int(len(certified)),
                "uncertified_partition_rows": int(len(group) - len(certified)),
                "donor_free_stable_rate": estimate,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
