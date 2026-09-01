"""Complete reproducible pipeline for the paper's simulation study.

This module contains the study design, resumable optimization passes,
canonical-data construction, selective retries, validation, tables, and
figures.  :func:`run_complete_paper_study` is the only public entry point the
notebook needs.

The design deliberately separates three sources of randomness:

* the compatibility-graph subsample depends only on base pool and pool size;
* the organizational partition also depends on organization count/replication;
* donor permutations depend on the market, but not on the algorithm or cycle cap.

Raw results are appended after every algorithm/order run, so an interrupted
study can resume without repeating completed work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import platform
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pulp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    lexicographic_floor_core_search,
    make_partition,
    prepare_lexicographic_floor_baseline,
    separate_blocking_coalition,
    strong_core_heuristic,
)


@dataclass(frozen=True)
class StudyConfig:
    instance_dir: str = "instances_large"
    output_dir: str = "results/management_science_full_core/reproduction_work/cap09"
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
    primary_num_players: Tuple[int, ...] = (5, 10, 15, 20, 30)
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
        Path("paper_simulations.py"),
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


def run_study(
    config: StudyConfig = StudyConfig(),
    *,
    active_num_players: Sequence[int] | None = None,
    primary_num_players: Sequence[int] | None = None,
) -> Dict[str, Path]:
    """Run or resume a scoped pass without recomputing checkpointed rows."""
    selected_paths = select_base_instances(config)
    manifest_path = write_study_manifest(config, selected_paths)
    output_dir = Path(config.output_dir)
    raw_path = output_dir / "raw_results.jsonl"
    csv_path = output_dir / "raw_results.csv"
    completed = _load_checkpoint(raw_path)

    active_players = tuple(active_num_players or config.num_players)
    if not set(active_players).issubset(config.num_players):
        raise ValueError("active_num_players must be a subset of the frozen design")
    primary_players = tuple(
        config.primary_num_players
        if primary_num_players is None
        else primary_num_players
    )
    if not set(primary_players).issubset(active_players):
        raise ValueError("primary_num_players must be a subset of active_num_players")

    total_cells = (
        len(selected_paths)
        * len(config.pool_sizes)
        * len(active_players)
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

            for num_players in active_players:
                run_primary = num_players in primary_players
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
                            inferred_algorithms = (
                                ("strong", "weak") if run_primary else ("strong",)
                            )
                            for algorithm in inferred_algorithms:
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

                            if run_primary and not donor_free_strong_rows:
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

                        if not run_primary:
                            elapsed_study = perf_counter() - study_started
                            rate = cells_seen / elapsed_study if elapsed_study else 0.0
                            remaining = (
                                (cell_limit - cells_seen) / rate if rate else float("inf")
                            )
                            print(
                                f"[{cells_seen}/{cell_limit}] {market_id} | "
                                f"elapsed={elapsed_study / 60:.1f} min | "
                                f"ETA={remaining / 60:.1f} min"
                            )
                            continue

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


# ---------------------------------------------------------------------------
# Conditional floor-preserving lexicographic stage
# ---------------------------------------------------------------------------

ALGORITHM = "lexicographic_floor_stabilization"
EVIDENCE_TYPE = "lexicographic_floors_aggregate_cut_heuristic_exact_final_check"
AUGMENTATION_ID = "lexicographic_frozen_floors_v1"
MANIFEST_NAME = "lexicographic_floor_augmentation_manifest.json"
SUMMARY_NAME = "lexicographic_floor_summary.csv"
CODE_FILES = (
    Path("KEP_functions.py"),
    Path("paper_simulations.py"),
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_key(row: Mapping[str, object]) -> Tuple[str, str, int]:
    return (
        str(row["market_id"]),
        str(row["algorithm"]),
        int(row.get("order_rep", 0)),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _source_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("algorithm")) != ALGORITHM]


def _canonical_rows_hash(rows: Iterable[Mapping[str, object]]) -> str:
    digest = sha256()
    ordered = sorted((dict(row) for row in rows), key=_result_key)
    for row in ordered:
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _target_initial_rows(
    rows: Iterable[Mapping[str, object]],
) -> Dict[str, dict[str, object]]:
    targets: Dict[str, dict[str, object]] = {}
    for original in rows:
        row = dict(original)
        if (
            row.get("algorithm") == "lexicographic_initial"
            and row.get("certified") is True
            and row.get("in_core") is False
        ):
            market_id = str(row["market_id"])
            if market_id in targets:
                raise ValueError(f"Duplicate challenged initial row for {market_id}")
            targets[market_id] = row
    return targets


def _code_hashes() -> dict[str, str]:
    missing = [str(path) for path in CODE_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing augmentation code files: {missing}")
    return {path.as_posix(): _sha256_file(path) for path in CODE_FILES}


def _load_study_manifest(output_dir: Path) -> dict[str, object]:
    path = output_dir / "study_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_instances(study_manifest: Mapping[str, object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for record in study_manifest["instances"]:
        path = Path(str(record["path"]))
        if not path.exists():
            raise FileNotFoundError(path)
        actual = _sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"Frozen instance hash mismatch for {path}")
        paths[str(record["filename"])] = path
    return paths


def _augmentation_manifest_payload(
    output_dir: Path,
    study_manifest: Mapping[str, object],
    source_rows: Sequence[Mapping[str, object]],
    targets: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    config = dict(study_manifest["config"])
    return {
        "augmentation_id": AUGMENTATION_ID,
        "algorithm": ALGORITHM,
        "evidence_type": EVIDENCE_TYPE,
        "output_dir": output_dir.as_posix(),
        "source_rows_fingerprint": _canonical_rows_hash(source_rows),
        "target_market_ids": sorted(targets),
        "target_markets": len(targets),
        "maximum_order_rows": len(targets) * int(config["donor_order_reps"]),
        "study_config": config,
        "instance_hashes": {
            str(record["filename"]): str(record["sha256"])
            for record in study_manifest["instances"]
        },
        "code_hashes": _code_hashes(),
        "method": {
            "baseline": "donor-free four-tier lexicographic optimum",
            "constraints": "all four donor-free tier values retained as lower bounds",
            "master_objective": "minimize donors used",
            "search": "aggregate weak-core cuts; exact final weak-core separator",
            "donor_rule": "introduce next seeded donor only after floor master infeasibility",
            "fourth_tier_reference": "recipient hardness frozen on donor-free graph",
            "interpretation": "positive existence evidence; failure does not prove emptiness",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": None,
    }


def _load_or_create_augmentation_manifest(
    output_dir: Path,
    study_manifest: Mapping[str, object],
    source_rows: Sequence[Mapping[str, object]],
    targets: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    path = output_dir / MANIFEST_NAME
    expected = _augmentation_manifest_payload(
        output_dir, study_manifest, source_rows, targets
    )
    if not path.exists():
        path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        return expected
    existing = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "augmentation_id",
        "algorithm",
        "evidence_type",
        "source_rows_fingerprint",
        "target_market_ids",
        "study_config",
        "instance_hashes",
        "code_hashes",
        "method",
    ):
        if existing.get(field) != expected.get(field):
            raise ValueError(
                f"Existing {path} is incompatible in field {field!r}. "
                "Do not edit backend or runner code after an augmentation starts."
            )
    return existing


def _base_fields(initial_row: Mapping[str, object]) -> dict[str, object]:
    excluded = {
        "algorithm",
        "order_rep",
        "evidence_type",
        "in_core",
        "certified",
        "termination_reason",
        "solver_status",
        "donors_introduced",
        "donors_used",
        "introduced_donor_ids",
        "used_donor_ids",
        "donor_order",
        "real_transplants",
        "baseline_real_transplants",
        "runtime_seconds",
        "runner_seconds",
        "cycle_enumeration_seconds",
        "optimization_seconds",
        "separation_seconds",
        "num_master_solves",
        "num_optimization_solves",
        "num_separation_solves",
        "num_cuts",
        "max_mip_gap",
        "blocked_k2",
        "blocked_k3",
        "blocked_k4",
        "cap_sensitivity_certified",
        "cap_sensitivity_seconds",
        "repair_id",
    }
    return {key: value for key, value in initial_row.items() if key not in excluded}


def _augmentation_row(
    initial_row: Mapping[str, object],
    result: Mapping[str, object],
    order_rep: int,
    runner_seconds: float,
    baseline_seconds_charged: float,
) -> dict[str, object]:
    row = _flatten_result(
        _base_fields(initial_row),
        algorithm=ALGORITHM,
        order_rep=order_rep,
        evidence_type=EVIDENCE_TYPE,
        result=result,
        runner_seconds=runner_seconds,
    )
    row.update(
        {
            "augmentation_id": AUGMENTATION_ID,
            "source_initial_algorithm": "lexicographic_initial",
            "source_initial_order_rep": -1,
            "source_initial_certified": bool(initial_row.get("certified")),
            "source_initial_in_core": bool(initial_row.get("in_core")),
            "baseline_objective_tiers": dict(
                result.get("baseline_objective_tiers", {})
            ),
            "achieved_objective_tiers": dict(result.get("objective_tiers", {})),
            "objective_floor_slacks": dict(
                result.get("objective_floor_slacks", {})
            ),
            "objective_floor_type": result.get("objective_floor_type"),
            "objective_score_reference": result.get("objective_score_reference"),
            "search_cut_type": result.get("search_cut_type"),
            "baseline_preparation_seconds": float(
                result.get("baseline_preparation_seconds", 0.0)
            ),
            "baseline_seconds_charged_to_runner": float(baseline_seconds_charged),
            "search_runtime_seconds": float(result.get("runtime_seconds", 0.0)),
        }
    )
    return row


def _write_raw_csv(raw_path: Path, csv_path: Path) -> None:
    rows = _read_jsonl(raw_path)
    keys = [_result_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Refusing to write CSV because raw JSONL has duplicate keys")
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_summary(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    frame = pd.DataFrame(
        [dict(row) for row in rows if str(row.get("algorithm")) == ALGORITHM]
    )
    summary_path = output_dir / SUMMARY_NAME
    if frame.empty:
        pd.DataFrame().to_csv(summary_path, index=False)
        return summary_path
    successful = frame[frame["certified"].eq(True) & frame["in_core"].eq(True)]
    market = successful.groupby("market_id", as_index=False).agg(
        successful_orders=("order_rep", "nunique"),
        best_introduced=("donors_introduced", "min"),
        worst_introduced=("donors_introduced", "max"),
        best_used=("donors_used", "min"),
        worst_used=("donors_used", "max"),
    )
    first = frame.sort_values(["market_id", "order_rep"]).drop_duplicates("market_id")
    base_columns = [
        "market_id",
        "instance",
        "pool_size",
        "num_players",
        "partition_rep",
        "Delta",
    ]
    result = first[base_columns].merge(market, on="market_id", how="left")
    result["has_certified_stable_result"] = result["successful_orders"].notna()
    result.to_csv(summary_path, index=False)
    return summary_path


def append_lexicographic_floor_results(
    output_dir: str | Path,
    *,
    expected_target_markets: int | None = None,
) -> dict[str, object]:
    """Run or resume the selective augmentation for one study folder."""
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_results.jsonl"
    csv_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    all_rows = _read_jsonl(raw_path)
    source_rows = _source_rows(all_rows)
    targets = _target_initial_rows(source_rows)
    if expected_target_markets is not None and len(targets) != expected_target_markets:
        raise ValueError(
            f"{output_dir}: expected {expected_target_markets} challenged markets, "
            f"found {len(targets)}"
        )
    study_manifest = _load_study_manifest(output_dir)
    config = dict(study_manifest["config"])
    instance_paths = _validate_instances(study_manifest)
    augmentation_manifest = _load_or_create_augmentation_manifest(
        output_dir, study_manifest, source_rows, targets
    )

    existing = {_result_key(row): row for row in all_rows}
    if len(existing) != len(all_rows):
        raise ValueError(f"{raw_path} contains duplicate result keys")
    graph_cache: dict[tuple[str, int], tuple[object, object, object, object, object]] = {}
    new_rows = 0
    markets_completed = 0
    started = perf_counter()

    for market_index, market_id in enumerate(sorted(targets), start=1):
        initial_row = targets[market_id]
        instance_name = str(initial_row["instance"])
        pool_size = int(initial_row["pool_size"])
        num_players = int(initial_row["num_players"])
        partition_rep = int(initial_row["partition_rep"])
        delta = int(initial_row["Delta"])
        cache_key = (instance_name, pool_size)
        if cache_key not in graph_cache:
            instance = load_instance(instance_paths[instance_name])
            features = build_graph_features(instance)
            graph_seed = int(initial_row["graph_seed"])
            vertices, adj_out, _, altruist_edges = build_compat_graph(
                instance,
                num_patients=pool_size,
                rng=np.random.default_rng(graph_seed),
            )
            if len(vertices) != int(initial_row["actual_vertices"]):
                raise ValueError(f"Vertex-count mismatch for {market_id}")
            if len(altruist_edges) != int(initial_row["available_altruists"]):
                raise ValueError(f"Altruist-count mismatch for {market_id}")
            graph_cache[cache_key] = (
                instance,
                features,
                vertices,
                adj_out,
                altruist_edges,
            )
        _, features, vertices, adj_out, altruist_edges = graph_cache[cache_key]
        partition = make_partition(
            vertices,
            num_players=num_players,
            var_size=int(config["partition_var_size"]),
            rng=np.random.default_rng(int(initial_row["partition_seed"])),
        )
        donor_orders = make_donor_orders(
            altruist_edges,
            master_seed=int(config["master_seed"]),
            instance_name=instance_name,
            pool_size=pool_size,
            num_players=num_players,
            partition_rep=partition_rep,
            repetitions=int(config["donor_order_reps"]),
        )
        solver_seed = stable_seed(
            int(config["master_seed"]),
            "solver",
            instance_name,
            pool_size,
            num_players,
            partition_rep,
            delta,
        )

        order0_key = (market_id, ALGORITHM, 0)
        order0 = existing.get(order0_key)
        if order0 is not None and bool(order0.get("certified")) and bool(
            order0.get("in_core")
        ) and int(order0.get("donors_introduced") or 0) == 0:
            markets_completed += 1
            print(f"[{market_index}/{len(targets)}] {market_id}: donor-free row already complete")
            continue

        baseline_started = perf_counter()
        baseline = prepare_lexicographic_floor_baseline(
            vertices,
            adj_out,
            partition,
            delta,
            features,
            max_coal_size=int(config["max_coal_size"]),
            solver=str(config["solver"]),
            time_limit=config.get("time_limit_seconds"),
            mip_gap=config.get("mip_gap"),
            threads=config.get("solver_threads"),
            solver_seed=stable_seed(solver_seed, "lex_initial_solver"),
        )
        baseline_elapsed = perf_counter() - baseline_started
        recorded_transplants = initial_row.get("real_transplants")
        recomputed_transplants = baseline["objective_tiers"].get("transplants")
        if (
            recorded_transplants is not None
            and recomputed_transplants is not None
            and abs(float(recorded_transplants) - float(recomputed_transplants)) > 1e-7
        ):
            raise ValueError(
                f"Donor-free lexicographic transplant tier mismatch for {market_id}: "
                f"recorded={recorded_transplants}, recomputed={recomputed_transplants}"
            )

        orders_to_run = range(int(config["donor_order_reps"]))
        for order_rep in orders_to_run:
            key = (market_id, ALGORITHM, order_rep)
            if key in existing:
                continue
            search_started = perf_counter()
            result = lexicographic_floor_core_search(
                vertices,
                adj_out,
                partition,
                delta,
                features,
                max_coal_size=int(config["max_coal_size"]),
                solver=str(config["solver"]),
                altruist_edges=altruist_edges,
                rng=np.random.default_rng(stable_seed(solver_seed, "lex", order_rep)),
                donor_order=donor_orders[order_rep],
                time_limit=config.get("time_limit_seconds"),
                mip_gap=config.get("mip_gap"),
                threads=config.get("solver_threads"),
                solver_seed=stable_seed(solver_seed, "lex_solver", order_rep),
                baseline=baseline,
            )
            search_elapsed = perf_counter() - search_started
            baseline_charge = baseline_elapsed if order_rep == 0 else 0.0
            row = _augmentation_row(
                initial_row,
                result,
                order_rep,
                runner_seconds=search_elapsed + baseline_charge,
                baseline_seconds_charged=baseline_charge,
            )
            _append_jsonl(raw_path, row)
            existing[key] = row
            new_rows += 1
            print(
                f"[{market_index}/{len(targets)}] {market_id} order={order_rep}: "
                f"certified={row['certified']} stable={row['in_core']} "
                f"introduced={row['donors_introduced']} used={row['donors_used']}"
            )
            if (
                order_rep == 0
                and bool(row.get("certified"))
                and bool(row.get("in_core"))
                and int(row.get("donors_introduced") or 0) == 0
            ):
                break

        markets_completed += 1

    final_rows = _read_jsonl(raw_path)
    _write_raw_csv(raw_path, csv_path)
    summary_path = _write_summary(output_dir, final_rows)
    validation = validate_lexicographic_floor_results(
        output_dir,
        expected_target_markets=expected_target_markets,
    )
    manifest_path = output_dir / MANIFEST_NAME
    completed_manifest = dict(augmentation_manifest)
    completed_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    completed_manifest["validation"] = validation
    manifest_path.write_text(
        json.dumps(completed_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": output_dir,
        "target_markets": len(targets),
        "markets_processed": markets_completed,
        "new_rows": new_rows,
        "total_augmentation_rows": validation["augmentation_rows"],
        "certified_stable_markets": validation["certified_stable_markets"],
        "unresolved_markets": validation["unresolved_markets"],
        "elapsed_seconds": perf_counter() - started,
        "raw_jsonl": raw_path,
        "raw_csv": csv_path,
        "summary_csv": summary_path,
        "augmentation_manifest": manifest_path,
    }


def validate_lexicographic_floor_results(
    output_dir: str | Path,
    *,
    expected_target_markets: int | None = None,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Validate selective scope, conditional completeness, and floor claims."""
    output_dir = Path(output_dir)
    rows = _read_jsonl(output_dir / "raw_results.jsonl")
    source = _source_rows(rows)
    targets = _target_initial_rows(source)
    if expected_target_markets is not None and len(targets) != expected_target_markets:
        raise ValueError(
            f"Expected {expected_target_markets} targets, found {len(targets)}"
        )
    augmentation = [row for row in rows if row.get("algorithm") == ALGORITHM]
    keys = [_result_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Raw results contain duplicate keys")
    unknown_markets = sorted(
        set(str(row["market_id"]) for row in augmentation).difference(targets)
    )
    if unknown_markets:
        raise ValueError(f"Augmentation contains nontarget markets: {unknown_markets}")

    by_market: dict[str, list[dict[str, object]]] = {}
    for row in augmentation:
        if row.get("augmentation_id") != AUGMENTATION_ID:
            raise ValueError(f"Wrong augmentation_id in {row['market_id']}")
        floors = dict(row.get("baseline_objective_tiers", {}))
        achieved = dict(row.get("achieved_objective_tiers", {}))
        if set(floors) != {"transplants", "cycle_count", "same_blood", "hard_match"}:
            raise ValueError(f"Missing floor tiers in {row['market_id']}")
        if set(achieved) != set(floors):
            raise ValueError(f"Missing achieved tiers in {row['market_id']}")
        if bool(row.get("certified")) and bool(row.get("in_core")):
            for name, floor in floors.items():
                if float(achieved[name]) + tolerance < float(floor):
                    raise ValueError(
                        f"Certified row violates {name} floor in {row['market_id']}"
                    )
        introduced = [int(v) for v in row.get("introduced_donor_ids", [])]
        used = [int(v) for v in row.get("used_donor_ids", [])]
        if len(introduced) != len(set(introduced)) or len(used) != len(set(used)):
            raise ValueError(f"Duplicate donor ids in {row['market_id']}")
        if not set(used).issubset(introduced):
            raise ValueError(f"Used donor was not introduced in {row['market_id']}")
        if len(introduced) != int(row.get("donors_introduced") or 0):
            raise ValueError(f"Introduced donor count mismatch in {row['market_id']}")
        if len(used) != int(row.get("donors_used") or 0):
            raise ValueError(f"Used donor count mismatch in {row['market_id']}")
        by_market.setdefault(str(row["market_id"]), []).append(row)

    incomplete: list[str] = []
    certified_stable = 0
    for market_id in targets:
        market_rows = sorted(by_market.get(market_id, []), key=lambda row: row["order_rep"])
        if not market_rows:
            incomplete.append(market_id)
            continue
        order_reps = [int(row["order_rep"]) for row in market_rows]
        order0 = next((row for row in market_rows if int(row["order_rep"]) == 0), None)
        if order0 is None:
            incomplete.append(market_id)
            continue
        donor_free = bool(order0.get("certified")) and bool(order0.get("in_core")) and int(
            order0.get("donors_introduced") or 0
        ) == 0
        expected_orders = [0] if donor_free else [0, 1, 2]
        if order_reps != expected_orders:
            incomplete.append(market_id)
        if any(bool(row.get("certified")) and bool(row.get("in_core")) for row in market_rows):
            certified_stable += 1
    if incomplete:
        raise ValueError(
            f"Augmentation is incomplete for {len(incomplete)} markets; "
            f"first={incomplete[:5]}"
        )

    return {
        "augmentation_id": AUGMENTATION_ID,
        "target_markets": len(targets),
        "augmentation_rows": len(augmentation),
        "certified_stable_markets": certified_stable,
        "unresolved_markets": len(targets) - certified_stable,
        "donor_free_tied_optimum_markets": sum(
            1
            for rows_for_market in by_market.values()
            if any(
                bool(row.get("certified"))
                and bool(row.get("in_core"))
                and int(row.get("donors_introduced") or 0) == 0
                for row in rows_for_market
            )
        ),
    }


# ---------------------------------------------------------------------------
# Full weak-core and lexicographic stage
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/management_science_full_core/reproduction_work/cap30")
RAW_NAME = "raw_results.jsonl"
CSV_NAME = "raw_results.csv"
PROTOCOL_NAME = "robustness_protocol.json"
WEAK_ALGORITHM = "weak_heuristic"
LEX_INITIAL_ALGORITHM = "lexicographic_initial"
LEX_FLOOR_ALGORITHM = "lexicographic_floor_stabilization"


def robustness_config(output_dir: str | Path = OUTPUT_DIR) -> StudyConfig:
    """Return the frozen cap-30 design derived from the main study."""
    return StudyConfig(
        instance_dir="instances_large",
        output_dir=str(output_dir),
        num_base_instances=30,
        instance_selection_seed=20260819,
        master_seed=20260819,
        pool_sizes=(100, 200, 500),
        num_players=(20, 30),
        deltas=(2, 3),
        partition_reps=2,
        partition_var_size=1,
        donor_order_reps=3,
        max_coal_size=30,
        solver="GUROBI",
        time_limit_seconds=200,
        mip_gap=0.0,
        solver_threads=16,
        run_heuristic_diagnostics=True,
        max_cells=None,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_manifests(config: StudyConfig) -> list[Path]:
    selected = select_base_instances(config)
    write_study_manifest(config, selected)
    reference_path = Path(config.output_dir).parent / "cap09" / "study_manifest.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_names = [str(row["filename"]) for row in reference["instances"]]
        selected_names = [path.name for path in selected]
        if selected_names != reference_names:
            raise ValueError("Cap-30 base-instance selection differs from cap 9")
        reference_hashes = {
            str(row["filename"]): str(row["sha256"])
            for row in reference["instances"]
        }
        for path in selected:
            if _sha256_file(path) != reference_hashes[path.name]:
                raise ValueError(f"Instance hash differs from cap 9: {path}")

    protocol_path = Path(config.output_dir) / PROTOCOL_NAME
    payload = {
        "protocol_id": "management_science_full_core_n20_n30_v1",
        "max_coal_size": 30,
        "num_players": [20, 30],
        "market_cells": 720,
        "weak_method": (
            "aggregate-cut weak-core heuristic with exact final full-core audit; "
            "failure does not prove emptiness"
        ),
        "lex_method": (
            "donor-free lexicographic audit followed, only if blocked, by "
            "frozen-floor aggregate-cut stabilization with exact final audit"
        ),
        "conditional_orders": (
            "run order zero first; skip orders one and two after a certified "
            "donor-free stable result"
        ),
        "seed_reference": "sibling cap09 checkpoint",
        "code_hashes": {
            path.as_posix(): _sha256_file(path)
            for path in (
                Path("KEP_functions.py"),
                Path("paper_simulations.py"),
            )
        },
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing {protocol_path} differs from the current protocol. "
                "Do not edit code after the cap-30 study starts."
            )
    else:
        protocol_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return selected


def _total_markets(config: StudyConfig) -> int:
    return (
        config.num_base_instances
        * len(config.pool_sizes)
        * len(config.num_players)
        * config.partition_reps
        * len(config.deltas)
    )


def _iter_markets(config: StudyConfig) -> Iterator[dict[str, object]]:
    selected_paths = _prepare_manifests(config)
    for instance_path in selected_paths:
        load_started = perf_counter()
        instance = load_instance(instance_path)
        instance_load_seconds = perf_counter() - load_started
        features = build_graph_features(instance)
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
                        base["max_coal_size"] = 30
                        yield {
                            "base": base,
                            "vertices": vertices,
                            "adj_out": adj_out,
                            "partition": partition,
                            "features": features,
                            "altruist_edges": altruist_edges,
                            "donor_orders": donor_orders,
                            "solver_seed": solver_seed,
                        }


def _load_state(config: StudyConfig) -> tuple[Path, Path, Dict[Tuple[str, str, int], dict[str, object]]]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    csv_path = output_dir / CSV_NAME
    completed = _load_checkpoint(raw_path)
    return raw_path, csv_path, completed


def _save(
    raw_path: Path,
    completed: Dict[Tuple[str, str, int], dict[str, object]],
    row: dict[str, object],
) -> dict[str, object]:
    key = _result_key(row)
    if key not in completed:
        _append_checkpoint(raw_path, row)
        completed[key] = row
    return completed[key]


def _write_csv(csv_path: Path, completed: Mapping[Tuple[str, str, int], Mapping[str, object]]) -> None:
    pd.DataFrame(completed.values()).to_csv(csv_path, index=False)


def _weak_summary(config: StudyConfig, completed: Mapping[Tuple[str, str, int], Mapping[str, object]]) -> Path:
    output_path = Path(config.output_dir) / "weak_full_core_summary.csv"
    frame = pd.DataFrame(
        [row for row in completed.values() if row.get("algorithm") == WEAK_ALGORITHM]
    )
    if frame.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return output_path
    success = frame[frame["certified"].eq(True) & frame["in_core"].eq(True)]
    market = success.groupby("market_id", as_index=False).agg(
        successful_orders=("order_rep", "nunique"),
        best_introduced=("donors_introduced", "min"),
        best_used=("donors_used", "min"),
    )
    base = frame.sort_values(["market_id", "order_rep"]).drop_duplicates("market_id")
    result = base[
        ["market_id", "instance", "pool_size", "num_players", "partition_rep", "Delta"]
    ].merge(market, on="market_id", how="left")
    result["has_certified_stable_result"] = result["successful_orders"].notna()
    result.to_csv(output_path, index=False)
    return output_path


def run_weak_full_core_robustness(
    config: StudyConfig | None = None,
) -> dict[str, object]:
    """Run/resume only the weak-core heuristic at full coalition cap."""
    config = config or robustness_config()
    if tuple(config.num_players) != (20, 30) or config.max_coal_size != 30:
        raise ValueError("Weak robustness design must use n=(20,30), cap=30")
    raw_path, csv_path, completed = _load_state(config)
    total = _total_markets(config)
    started = perf_counter()
    new_rows = 0

    for market_index, cell in enumerate(_iter_markets(config), start=1):
        base = cell["base"]
        market_id = str(base["market_id"])
        order0 = completed.get((market_id, WEAK_ALGORITHM, 0))
        if order0 is not None and bool(order0.get("certified")) and bool(
            order0.get("in_core")
        ) and int(order0.get("donors_introduced") or 0) == 0:
            print(f"[{market_index}/{total}] {market_id}: weak donor-free complete")
            continue
        if order0 is not None and all(
            (market_id, WEAK_ALGORITHM, order_rep) in completed
            for order_rep in range(config.donor_order_reps)
        ):
            print(f"[{market_index}/{total}] {market_id}: weak order study complete")
            continue

        for order_rep in range(config.donor_order_reps):
            key = (market_id, WEAK_ALGORITHM, order_rep)
            if key in completed:
                continue
            result, elapsed = _run_call(
                core_heuristic,
                cell["vertices"],
                cell["adj_out"],
                cell["partition"],
                int(base["Delta"]),
                solver=config.solver,
                max_coal_size=config.max_coal_size,
                max_altruists=len(cell["altruist_edges"]),
                rng=np.random.default_rng(
                    stable_seed(cell["solver_seed"], "weak", order_rep)
                ),
                altruist_edges=cell["altruist_edges"],
                donor_order=cell["donor_orders"][order_rep],
                time_limit=config.time_limit_seconds,
                mip_gap=config.mip_gap,
                threads=config.solver_threads,
                solver_seed=stable_seed(
                    cell["solver_seed"], "weak_solver", order_rep
                ),
            )
            result = dict(result)
            result["certified"] = bool(result.get("final_in_core"))
            result["termination_reason"] = (
                "heuristic_found_full_weak_core_allocation"
                if result.get("final_in_core")
                else "heuristic_did_not_find_full_weak_core_allocation"
            )
            row = _flatten_result(
                base,
                algorithm=WEAK_ALGORITHM,
                order_rep=order_rep,
                evidence_type="aggregate_cut_heuristic_exact_full_core_final_check",
                result=result,
                runner_seconds=elapsed,
            )
            row["max_coal_size"] = 30
            _save(raw_path, completed, row)
            new_rows += 1
            print(
                f"[{market_index}/{total}] {market_id} weak order={order_rep}: "
                f"stable={row['in_core']} introduced={row['donors_introduced']}"
            )
            if (
                order_rep == 0
                and bool(row.get("certified"))
                and bool(row.get("in_core"))
                and int(row.get("donors_introduced") or 0) == 0
            ):
                break

    _write_csv(csv_path, completed)
    summary_path = _weak_summary(config, completed)
    validation = validate_cap30_results(config)
    return {
        "procedure": "weak_heuristic",
        "markets": total,
        "new_rows": new_rows,
        "elapsed_seconds": perf_counter() - started,
        "raw_jsonl": raw_path,
        "raw_csv": csv_path,
        "summary_csv": summary_path,
        **validation["weak"],
    }


def _baseline_result(
    baseline: Mapping[str, object], donor_order: list[int]
) -> dict[str, object]:
    certified = bool(baseline.get("certified"))
    in_core = bool(baseline.get("in_core"))
    if certified and in_core:
        reason = "stable_donor_free_lexicographic_allocation"
    elif certified:
        reason = "blocked_donor_free_lexicographic_allocation"
    else:
        reason = "donor_free_lexicographic_audit_not_certified"
    return {
        "solution": set(baseline.get("selection", set())),
        "final_in_core": in_core,
        "in_core": in_core,
        "certified": certified,
        "termination_reason": reason,
        "solver_status": baseline.get("solver_status"),
        "objective_tiers": dict(baseline.get("objective_tiers", {})),
        "altruists_introduced": 0,
        "altruists_added": 0,
        "altruists_used": 0,
        "introduced_donor_ids": [],
        "used_donor_ids": [],
        "donor_order": list(donor_order),
        "runtime_seconds": float(baseline.get("runtime_seconds", 0.0)),
        "cycle_enumeration_seconds": baseline.get("cycle_enumeration_seconds"),
        "optimization_seconds": baseline.get("optimization_seconds"),
        "separation_seconds": baseline.get("separation_seconds"),
        "num_optimization_solves": baseline.get("num_optimization_solves"),
        "num_separation_solves": baseline.get("num_separation_solves"),
        "num_cuts": 0,
        "max_mip_gap": None,
    }


def _floor_row(
    base: Mapping[str, object],
    result: Mapping[str, object],
    order_rep: int,
    elapsed: float,
) -> dict[str, object]:
    row = _flatten_result(
        base,
        algorithm=LEX_FLOOR_ALGORITHM,
        order_rep=order_rep,
        evidence_type="lexicographic_floors_aggregate_cut_heuristic_exact_full_core_final_check",
        result=result,
        runner_seconds=elapsed,
    )
    row.update(
        {
            "max_coal_size": 30,
            "baseline_objective_tiers": dict(
                result.get("baseline_objective_tiers", {})
            ),
            "achieved_objective_tiers": dict(result.get("objective_tiers", {})),
            "objective_floor_slacks": dict(
                result.get("objective_floor_slacks", {})
            ),
            "objective_floor_type": result.get("objective_floor_type"),
            "objective_score_reference": result.get("objective_score_reference"),
            "search_cut_type": result.get("search_cut_type"),
        }
    )
    return row


def _lex_summary(config: StudyConfig, completed: Mapping[Tuple[str, str, int], Mapping[str, object]]) -> Path:
    output_path = Path(config.output_dir) / "lexicographic_full_core_summary.csv"
    initial = pd.DataFrame(
        [row for row in completed.values() if row.get("algorithm") == LEX_INITIAL_ALGORITHM]
    )
    floor = pd.DataFrame(
        [row for row in completed.values() if row.get("algorithm") == LEX_FLOOR_ALGORITHM]
    )
    if initial.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return output_path
    initial = initial[
        [
            "market_id",
            "instance",
            "pool_size",
            "num_players",
            "partition_rep",
            "Delta",
            "certified",
            "in_core",
        ]
    ].rename(
        columns={"certified": "initial_certified", "in_core": "initial_in_core"}
    )
    if floor.empty:
        initial.to_csv(output_path, index=False)
        return output_path
    success = floor[floor["certified"].eq(True) & floor["in_core"].eq(True)]
    assisted = success.groupby("market_id", as_index=False).agg(
        successful_orders=("order_rep", "nunique"),
        best_introduced=("donors_introduced", "min"),
        best_used=("donors_used", "min"),
    )
    result = initial.merge(assisted, on="market_id", how="left")
    result.to_csv(output_path, index=False)
    return output_path


def run_lexicographic_full_core_robustness(
    config: StudyConfig | None = None,
) -> dict[str, object]:
    """Run/resume the cap-30 lex audit and conditional floor stabilization."""
    config = config or robustness_config()
    if tuple(config.num_players) != (20, 30) or config.max_coal_size != 30:
        raise ValueError("Lex robustness design must use n=(20,30), cap=30")
    raw_path, csv_path, completed = _load_state(config)
    total = _total_markets(config)
    started = perf_counter()
    new_rows = 0

    for market_index, cell in enumerate(_iter_markets(config), start=1):
        base = cell["base"]
        market_id = str(base["market_id"])
        initial_key = (market_id, LEX_INITIAL_ALGORITHM, -1)
        initial_row = completed.get(initial_key)
        if initial_row is not None:
            if bool(initial_row.get("certified")) and bool(initial_row.get("in_core")):
                print(f"[{market_index}/{total}] {market_id}: initial lex stable")
                continue
            if not bool(initial_row.get("certified")):
                print(f"[{market_index}/{total}] {market_id}: initial lex unresolved")
                continue
            order0 = completed.get((market_id, LEX_FLOOR_ALGORITHM, 0))
            if order0 is not None and bool(order0.get("certified")) and bool(
                order0.get("in_core")
            ) and int(order0.get("donors_introduced") or 0) == 0:
                print(f"[{market_index}/{total}] {market_id}: stable tied optimum complete")
                continue
            if order0 is not None and all(
                (market_id, LEX_FLOOR_ALGORITHM, order_rep) in completed
                for order_rep in range(config.donor_order_reps)
            ):
                print(f"[{market_index}/{total}] {market_id}: lex order study complete")
                continue

        baseline_started = perf_counter()
        baseline = prepare_lexicographic_floor_baseline(
            cell["vertices"],
            cell["adj_out"],
            cell["partition"],
            int(base["Delta"]),
            cell["features"],
            max_coal_size=config.max_coal_size,
            solver=config.solver,
            time_limit=config.time_limit_seconds,
            mip_gap=config.mip_gap,
            threads=config.solver_threads,
            solver_seed=stable_seed(cell["solver_seed"], "lex_initial_solver"),
        )
        baseline_elapsed = perf_counter() - baseline_started
        if initial_row is not None:
            stored_state = (
                bool(initial_row.get("certified")),
                bool(initial_row.get("in_core")),
            )
            rebuilt_state = (
                bool(baseline.get("certified")),
                bool(baseline.get("in_core")),
            )
            if stored_state != rebuilt_state:
                raise ValueError(
                    f"Recomputed lexicographic baseline differs for {market_id}: "
                    f"stored={stored_state}, rebuilt={rebuilt_state}. The frozen "
                    "code, solver settings, or source instance may have changed."
                )
        if initial_row is None:
            initial_result = _baseline_result(baseline, cell["donor_orders"][0])
            initial_row = _flatten_result(
                base,
                algorithm=LEX_INITIAL_ALGORITHM,
                order_rep=-1,
                evidence_type="exact_selected_allocation_full_weak_core_check",
                result=initial_result,
                runner_seconds=baseline_elapsed,
            )
            initial_row["max_coal_size"] = 30
            _save(raw_path, completed, initial_row)
            new_rows += 1
            print(
                f"[{market_index}/{total}] {market_id} initial lex: "
                f"certified={initial_row['certified']} stable={initial_row['in_core']}"
            )

        if not bool(baseline.get("certified")) or bool(baseline.get("in_core")):
            continue

        for order_rep in range(config.donor_order_reps):
            key = (market_id, LEX_FLOOR_ALGORITHM, order_rep)
            if key in completed:
                continue
            result, elapsed = _run_call(
                lexicographic_floor_core_search,
                cell["vertices"],
                cell["adj_out"],
                cell["partition"],
                int(base["Delta"]),
                cell["features"],
                max_coal_size=config.max_coal_size,
                solver=config.solver,
                altruist_edges=cell["altruist_edges"],
                rng=np.random.default_rng(
                    stable_seed(cell["solver_seed"], "lex", order_rep)
                ),
                donor_order=cell["donor_orders"][order_rep],
                time_limit=config.time_limit_seconds,
                mip_gap=config.mip_gap,
                threads=config.solver_threads,
                solver_seed=stable_seed(
                    cell["solver_seed"], "lex_solver", order_rep
                ),
                baseline=baseline,
            )
            row = _floor_row(base, result, order_rep, elapsed)
            _save(raw_path, completed, row)
            new_rows += 1
            print(
                f"[{market_index}/{total}] {market_id} lex floor order={order_rep}: "
                f"certified={row['certified']} stable={row['in_core']} "
                f"introduced={row['donors_introduced']}"
            )
            if (
                order_rep == 0
                and bool(row.get("certified"))
                and bool(row.get("in_core"))
                and int(row.get("donors_introduced") or 0) == 0
            ):
                break

    _write_csv(csv_path, completed)
    summary_path = _lex_summary(config, completed)
    validation = validate_cap30_results(config)
    return {
        "procedure": "lexicographic_full_core",
        "markets": total,
        "new_rows": new_rows,
        "elapsed_seconds": perf_counter() - started,
        "raw_jsonl": raw_path,
        "raw_csv": csv_path,
        "summary_csv": summary_path,
        **validation["lexicographic"],
    }


def validate_cap30_results(config: StudyConfig | None = None) -> dict[str, object]:
    """Return conditional completeness and certification counts for both cells."""
    config = config or robustness_config()
    completed = _load_checkpoint(Path(config.output_dir) / RAW_NAME)
    expected_markets = _total_markets(config)
    weak_rows = [row for row in completed.values() if row.get("algorithm") == WEAK_ALGORITHM]
    weak_markets = set(str(row["market_id"]) for row in weak_rows)
    weak_success = {
        str(row["market_id"])
        for row in weak_rows
        if bool(row.get("certified")) and bool(row.get("in_core"))
    }
    weak_by_market: dict[str, dict[int, Mapping[str, object]]] = {}
    for row in weak_rows:
        weak_by_market.setdefault(str(row["market_id"]), {})[
            int(row["order_rep"])
        ] = row
    weak_order_complete = 0
    for rows in weak_by_market.values():
        order0 = rows.get(0)
        if order0 is None:
            continue
        donor_free_success = (
            bool(order0.get("certified"))
            and bool(order0.get("in_core"))
            and int(order0.get("donors_introduced") or 0) == 0
        )
        expected_orders = {0} if donor_free_success else set(
            range(config.donor_order_reps)
        )
        if set(rows) == expected_orders:
            weak_order_complete += 1
    initial_rows = [
        row for row in completed.values() if row.get("algorithm") == LEX_INITIAL_ALGORITHM
    ]
    initial_markets = set(str(row["market_id"]) for row in initial_rows)
    initial_stable = {
        str(row["market_id"])
        for row in initial_rows
        if bool(row.get("certified")) and bool(row.get("in_core"))
    }
    initial_blocked = {
        str(row["market_id"])
        for row in initial_rows
        if bool(row.get("certified")) and not bool(row.get("in_core"))
    }
    floor_rows = [
        row for row in completed.values() if row.get("algorithm") == LEX_FLOOR_ALGORITHM
    ]
    floor_success = {
        str(row["market_id"])
        for row in floor_rows
        if bool(row.get("certified")) and bool(row.get("in_core"))
    }
    floor_by_market: dict[str, dict[int, Mapping[str, object]]] = {}
    for row in floor_rows:
        floor_by_market.setdefault(str(row["market_id"]), {})[
            int(row["order_rep"])
        ] = row
    lex_conditionally_complete = 0
    for row in initial_rows:
        market_id = str(row["market_id"])
        if not bool(row.get("certified")) or bool(row.get("in_core")):
            if market_id not in floor_by_market:
                lex_conditionally_complete += 1
            continue
        rows = floor_by_market.get(market_id, {})
        order0 = rows.get(0)
        if order0 is None:
            continue
        donor_free_success = (
            bool(order0.get("certified"))
            and bool(order0.get("in_core"))
            and int(order0.get("donors_introduced") or 0) == 0
        )
        expected_orders = {0} if donor_free_success else set(
            range(config.donor_order_reps)
        )
        if set(rows) == expected_orders:
            lex_conditionally_complete += 1
    return {
        "expected_markets": expected_markets,
        "weak": {
            "completed_markets": len(weak_markets),
            "certified_stable_markets": len(weak_success),
            "unresolved_or_not_found_markets": len(weak_markets - weak_success),
            "conditionally_complete_markets": weak_order_complete,
            "design_complete": weak_order_complete == expected_markets,
        },
        "lexicographic": {
            "initial_completed_markets": len(initial_markets),
            "initial_stable_markets": len(initial_stable),
            "initial_blocked_markets": len(initial_blocked),
            "initial_unresolved_markets": len(initial_markets - initial_stable - initial_blocked),
            "floor_stable_markets": len(floor_success),
            "conditionally_complete_markets": lex_conditionally_complete,
            "design_complete": lex_conditionally_complete == expected_markets,
        },
    }


# ---------------------------------------------------------------------------
# Canonical data, retries, tables, figures, and one-call orchestration
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "management_science_full_core_v1"
MASTER_SEED = 20260819
SOURCE_CAP9 = Path(
    "results/management_science_full_core/reproduction_work/cap09/raw_results.jsonl"
)
SOURCE_CAP30 = Path(
    "results/management_science_full_core/reproduction_work/cap30/raw_results.jsonl"
)
CANONICAL_PATH = Path("results/management_science_full_core/results.jsonl")
FIGURE_DIR = Path("figures/full_core_simulations")
TABLE_DIR = Path("results/management_science_full_core/tables")
RETRY_PATH = Path("results/management_science_full_core/retry_checkpoint.jsonl")
PLAYERS = (5, 10, 20, 30)
POOLS = (100, 200, 500)
DELTAS = (2, 3)
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260826

FROZEN_INSTANCES = (
    ("genxml-0.xml", "6b9072afd185042c30d8866743d14381f73da671c3e27ef16eb7aaa5e8227eea"),
    ("genxml-2.xml", "f546c80f06ab449654a6722178cdddf90fc2113dc50e292d3558b7c33acacb73"),
    ("genxml-4.xml", "92e613c24c9080142958bab8eb185e05c6233ba2b9c68898b171cc67954c1cd1"),
    ("genxml-11.xml", "c4fff5dc463f1ce9f0985f2bda819b7afb059deca3698640f4ad46bd30160b59"),
    ("genxml-13.xml", "ac0e3c344133a0809cebb69eb77f5ff29080599dfd707bbdbf5f17eef4e8956c"),
    ("genxml-23.xml", "50554b018d0a948933885b518bb41131d2dcf407683d7cee4e58da390be9b12d"),
    ("genxml-24.xml", "e192a03a097bdc4bb72745de9bf8811d9f617c615000b7ebfb6193edaedb76e6"),
    ("genxml-25.xml", "c54dcce28762bf451cf09849ec07f5ebdf49ee08d229d4de75a1389fb923c395"),
    ("genxml-27.xml", "ddc2ea6475951cbac3947a2a2daf648de25e49f70d973ba573beafc45dd3622a"),
    ("genxml-36.xml", "83dea482b47042c19ffd0c8b31afb5a632d68b9eed8d08df0dc6d918e1ba4776"),
    ("genxml-37.xml", "9dbece7ba7ec236a26824c6846adb9dfada7237ab29a8ee8d06bac40fd65fd51"),
    ("genxml-39.xml", "65c81367a099dfc2c9b433c0d5a572e318a5d7b306aaf4b2eb45c5c7d9217867"),
    ("genxml-46.xml", "ec7bae5ff96866e2ac15e0fcdb0205d1888f19ff4866151dba814824cb587ba4"),
    ("genxml-48.xml", "07fe02525155fdcfcfaa3361197faa31ee59e409976c91900262dc680e12f4a8"),
    ("genxml-57.xml", "48798bc6bc8e8f8be35b999fa4426a1b5abd4d88c6ce07de4c262a854bdc1557"),
    ("genxml-58.xml", "d03e9a81b0d54b5c7354f667830aa6cc410723004feecb134edfa98b65a98e02"),
    ("genxml-60.xml", "aba3cc71ec1f129dd3db2e3ae6c98694ad2383846f60829114fae3053a8d4d57"),
    ("genxml-65.xml", "171c61a3e879157131de68686e6e2c869cd6a7709b0176b201cd87ed4c09cb06"),
    ("genxml-68.xml", "10f1a80c43a915fd5ee8b486b5bba757f550cf26de72ba49e2a785c197616b58"),
    ("genxml-70.xml", "adad3c6118df61951613f8c550dba5297930da5eaef5a97cb3987cddb86115e0"),
    ("genxml-71.xml", "a6203a42a71d96303157b09d9ab6a2821e3f10078ad16b6d9f3cc0ac6f69c33f"),
    ("genxml-80.xml", "4aaf19c39c03a7e6cdde048912e6d9fe34dd998d2f679ff2456caac9e768f329"),
    ("genxml-83.xml", "2eb49f20b6dcf65eeda5f66c86ebf854efd5a078cee10ed9857620a294dbf62c"),
    ("genxml-84.xml", "7ca0bb0d160e9dda97e91c97a161f8b272dab64ab0c715bf5a9e28561abc9a18"),
    ("genxml-88.xml", "aaa5e1816e8e884f7fc6e3918d86d9ac5d1c31725b284afe8bfe654069840e83"),
    ("genxml-89.xml", "228d4e69a20a53756060057ce00ffc7be3f38952abd0230dad3a978ec33d34b9"),
    ("genxml-91.xml", "74fd83aafac56576294828dcf56f86d22c326f882d1c4e7a7004ff1574c6257f"),
    ("genxml-92.xml", "60ab168baab805d736412b1835eec2b0ab37b444cca25a643f73d7c5976ca491"),
    ("genxml-93.xml", "5988d6e6ceb158f917ab90928785082c4a3dfda76dec0df832deba640ca8a253"),
    ("genxml-94.xml", "9a816ca9cb2a1cc3603faaaaa93a8133bfd3df9be51763c575fe6d25beef6ecf"),
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_instances(instance_dir: str | Path = "instances_large") -> list[Path]:
    root = Path(instance_dir)
    paths: list[Path] = []
    for filename, expected_hash in FROZEN_INSTANCES:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_hash:
            raise ValueError(f"Instance hash mismatch for {path}: {actual}")
        paths.append(path)
    return paths


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def _result_frame(path: Path) -> pd.DataFrame:
    rows = _read_jsonl(path)
    frame = pd.DataFrame(rows)
    key = ["market_id", "algorithm", "order_rep"]
    if frame.duplicated(key).any():
        raise ValueError(f"Duplicate result keys in {path}")
    return frame


def _initial_lexicographic_rows(raw: pd.DataFrame, expected: int) -> pd.DataFrame:
    explicit = raw[raw["algorithm"].eq("lexicographic_initial")].copy()
    explicit_ids = set(explicit["market_id"])
    ordinary = raw[
        raw["algorithm"].eq("lexicographic")
        & raw["order_rep"].eq(0)
        & raw["donors_introduced"].fillna(10**9).eq(0)
        & ~raw["market_id"].isin(explicit_ids)
    ].copy()
    result = pd.concat([explicit, ordinary], ignore_index=True)
    if len(result) != expected or result["market_id"].nunique() != expected:
        raise ValueError(
            f"Expected {expected} initial lexicographic rows, found {len(result)}"
        )
    return result


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _clean_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return value


def _canonical_record(
    row: Mapping[str, Any],
    *,
    source_path: Path,
    source_cap: int,
    analysis_role: str,
    procedure: str,
    stage: str,
    full_core: bool,
    full_core_basis: str,
) -> dict[str, Any]:
    record = {str(key): _clean_json_value(value) for key, value in row.items()}
    source_algorithm = str(record.get("algorithm"))
    record.update(
        {
            "record_type": "result_call",
            "schema_version": SCHEMA_VERSION,
            "analysis_role": analysis_role,
            "procedure": procedure,
            "stage": stage,
            "source_algorithm": source_algorithm,
            "source_file": source_path.as_posix(),
            "source_file_sha256": sha256_file(source_path),
            "effective_coalition_cap": int(source_cap),
            "full_core": bool(full_core),
            "full_core_basis": full_core_basis,
        }
    )
    record["canonical_key"] = "|".join(
        [
            analysis_role,
            procedure,
            stage,
            str(record["market_id"]),
            source_algorithm,
            str(int(record.get("order_rep") or 0)),
        ]
    )
    return record


def _full_core_basis(n: int, cap: int) -> str:
    if cap >= n:
        return "all_coalitions_tested"
    if cap == n - 1:
        return "all_proper_coalitions_tested; grand coalition ruled out by coverage floor"
    raise ValueError(f"Cap {cap} is not a full-core audit for n={n}")


def _choose_one_weak_witness(group: pd.DataFrame) -> pd.Series:
    candidates = group[
        group["certified"].eq(True)
        & group["in_core"].eq(True)
        & group["donors_introduced"].fillna(10**9).eq(0)
    ].copy()
    if candidates.empty:
        candidates = group[
            group["certified"].eq(True) & group["in_core"].eq(True)
        ].copy()
    if candidates.empty:
        raise ValueError(f"No certified weak-core witness for {group.iloc[0]['market_id']}")
    candidates["_priority"] = candidates["algorithm"].map(
        {"weak": 0, "weak_heuristic": 1}
    ).fillna(9)
    return candidates.sort_values(["donors_introduced", "_priority", "order_rep"]).iloc[0]


def consolidate_completed_results(
    output_path: str | Path = CANONICAL_PATH,
    *,
    source_cap9: str | Path = SOURCE_CAP9,
    source_cap30: str | Path = SOURCE_CAP30,
) -> dict[str, Any]:
    """Create the one-file final dataset from the completed checkpoints."""
    output_path = Path(output_path)
    source_cap9 = Path(source_cap9)
    source_cap30 = Path(source_cap30)
    validate_frozen_instances()
    raw9 = _result_frame(source_cap9)
    raw30 = _result_frame(source_cap30)
    initial9 = _initial_lexicographic_rows(raw9, 720)
    initial30 = _initial_lexicographic_rows(raw30, 720)

    records: list[dict[str, Any]] = []

    # Primary weak-core evidence.  Low-organization markets keep one exact
    # positive witness; cap-30 rows retain conditional order/seed repetitions.
    weak9 = raw9[
        raw9["num_players"].isin([5, 10])
        & raw9["algorithm"].isin(["weak", "weak_heuristic"])
    ]
    selected_weak9 = pd.DataFrame(
        [_choose_one_weak_witness(group) for _, group in weak9.groupby("market_id")]
    )
    if len(selected_weak9) != 720:
        raise ValueError("Low-organization weak evidence is not one row per market")
    for _, row in selected_weak9.iterrows():
        n = int(row["num_players"])
        records.append(
            _canonical_record(
                row,
                source_path=source_cap9,
                source_cap=9,
                analysis_role="primary_full_core",
                procedure="weak_core",
                stage="core_search",
                full_core=True,
                full_core_basis=_full_core_basis(n, min(9, n)),
            )
        )
    weak30 = raw30[raw30["algorithm"].eq("weak_heuristic")]
    for _, row in weak30.iterrows():
        n = int(row["num_players"])
        records.append(
            _canonical_record(
                row,
                source_path=source_cap30,
                source_cap=30,
                analysis_role="primary_full_core",
                procedure="weak_core",
                stage="core_search",
                full_core=True,
                full_core_basis=_full_core_basis(n, min(30, n)),
            )
        )

    # One initial lexicographic audit per market.  Any time-limited call remains
    # unresolved here and is handled by the selective long-limit pass below.
    primary_initial = pd.concat(
        [
            initial9[initial9["num_players"].isin([5, 10])],
            initial30,
        ],
        ignore_index=True,
    )
    if len(primary_initial) != 1_440 or primary_initial["market_id"].nunique() != 1_440:
        raise ValueError("Primary initial lexicographic sample is incomplete")
    for _, row in primary_initial.iterrows():
        n = int(row["num_players"])
        source_path = source_cap9 if n <= 10 else source_cap30
        cap = 9 if n <= 10 else 30
        records.append(
            _canonical_record(
                row,
                source_path=source_path,
                source_cap=cap,
                analysis_role="primary_full_core",
                procedure="lexicographic_rule",
                stage="initial_audit",
                full_core=True,
                full_core_basis=_full_core_basis(n, min(cap, n)),
            )
        )

    # Frozen-floor follow-up only.  Donor-assisted rows from the earlier
    # re-optimization procedure are intentionally excluded.
    floor = pd.concat(
        [
            raw9[
                raw9["algorithm"].eq("lexicographic_floor_stabilization")
                & raw9["num_players"].isin([5, 10])
            ],
            raw30[raw30["algorithm"].eq("lexicographic_floor_stabilization")],
        ],
        ignore_index=True,
    )
    blocked_ids = set(
        primary_initial.loc[
            primary_initial["certified"].eq(True)
            & ~primary_initial["in_core"].eq(True),
            "market_id",
        ]
    )
    floor = floor[floor["market_id"].isin(blocked_ids)]
    for _, row in floor.iterrows():
        n = int(row["num_players"])
        source_path = source_cap9 if n <= 10 else source_cap30
        cap = 9 if n <= 10 else 30
        records.append(
            _canonical_record(
                row,
                source_path=source_path,
                source_cap=cap,
                analysis_role="primary_full_core",
                procedure="lexicographic_rule",
                stage="floor_stabilization",
                full_core=True,
                full_core_basis=_full_core_basis(n, min(cap, n)),
            )
        )

    # Cap-nine TU and strong results are supplementary.  They are full-core
    # for n=5,10 and capped robustness evidence for n=20,30.
    for procedure, algorithms in (
        ("tu_core", ["tu"]),
        ("strong_core", ["strong", "strong_heuristic"]),
    ):
        subset = raw9[raw9["algorithm"].isin(algorithms)]
        for _, row in subset.iterrows():
            n = int(row["num_players"])
            full = n <= 10
            basis = (
                _full_core_basis(n, min(9, n))
                if full
                else "coalitions of at most nine organizations"
            )
            records.append(
                _canonical_record(
                    row,
                    source_path=source_cap9,
                    source_cap=9,
                    analysis_role="supplementary_robustness",
                    procedure=procedure,
                    stage="core_search",
                    full_core=full,
                    full_core_basis=basis,
                )
            )

    keys = [record["canonical_key"] for record in records]
    if len(keys) != len(set(keys)):
        duplicates = pd.Series(keys)[pd.Series(keys).duplicated()].tolist()[:5]
        raise ValueError(f"Duplicate canonical keys: {duplicates}")

    metadata = {
        "record_type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "master_seed": MASTER_SEED,
            "base_instances": 30,
            "pool_sizes": list(POOLS),
            "num_players": list(PLAYERS),
            "partition_repetitions": 2,
            "cycle_caps": list(DELTAS),
            "donor_order_repetitions": 3,
            "primary_coalition_caps": {"5": 9, "10": 9, "20": 30, "30": 30},
            "integer_lexicographic_tiers_normalized": True,
        },
        "frozen_instances": [
            {"filename": filename, "sha256": digest}
            for filename, digest in FROZEN_INSTANCES
        ],
        "source_files": {
            path.as_posix(): sha256_file(path)
            for path in (source_cap9, source_cap30)
        },
        "code_files": {
            path.as_posix(): sha256_file(path)
            for path in (
                Path("KEP_functions.py"),
                Path("paper_simulations.py"),
            )
            if path.exists()
        },
        "selection_notes": [
            "Initial lexicographic rows are donor-free audits only.",
            "Follow-up lexicographic rows use frozen objective floors only.",
            "Non-finite legacy diagnostics are stored as null.",
            "Uncertified calls remain unresolved.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
        for record in sorted(
            records,
            key=lambda row: (
                row["analysis_role"],
                row["procedure"],
                row["stage"],
                row["market_id"],
                row["source_algorithm"],
                int(row.get("order_rep") or 0),
            ),
        ):
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return validate_canonical_results(output_path)


def load_canonical_results(path: str | Path = CANONICAL_PATH) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = _read_jsonl(Path(path))
    metadata_rows = [row for row in rows if row.get("record_type") == "metadata"]
    if len(metadata_rows) != 1:
        raise ValueError("Canonical JSONL must contain exactly one metadata record")
    calls = pd.DataFrame(
        [row for row in rows if row.get("record_type") == "result_call"]
    )
    return metadata_rows[0], calls


def market_outcomes(path: str | Path = CANONICAL_PATH) -> pd.DataFrame:
    _, calls = load_canonical_results(path)
    primary = calls[calls["analysis_role"].eq("primary_full_core")]
    initial = primary[
        primary["procedure"].eq("lexicographic_rule")
        & primary["stage"].eq("initial_audit")
    ].set_index("market_id")
    rows: list[dict[str, Any]] = []
    for market_id, lex in initial.iterrows():
        weak = primary[
            primary["market_id"].eq(market_id)
            & primary["procedure"].eq("weak_core")
        ]
        weak_success = weak[weak["certified"].eq(True) & weak["in_core"].eq(True)]
        weak_free = bool(weak_success["donors_introduced"].fillna(10**9).eq(0).any())
        weak_known = not weak_success.empty

        initial_known = bool(lex["certified"])
        initial_stable = initial_known and bool(lex["in_core"])
        floor = primary[
            primary["market_id"].eq(market_id)
            & primary["procedure"].eq("lexicographic_rule")
            & primary["stage"].eq("floor_stabilization")
        ]
        floor_success = floor[
            floor["certified"].eq(True) & floor["in_core"].eq(True)
        ]
        if initial_stable:
            lex_known = True
            best_introduced = 0.0
            best_used = 0.0
        elif initial_known and not floor_success.empty:
            lex_known = True
            best_introduced = float(floor_success["donors_introduced"].min())
            best_used = float(floor_success["donors_used"].min())
        else:
            lex_known = False
            best_introduced = np.nan
            best_used = np.nan
        rows.append(
            {
                "market_id": market_id,
                "instance": str(lex["instance"]),
                "pool_size": int(lex["pool_size"]),
                "num_players": int(lex["num_players"]),
                "partition_rep": int(lex["partition_rep"]),
                "Delta": int(lex["Delta"]),
                "weak_known": weak_known,
                "weak_donor_free": weak_free,
                "weak_assisted": weak_known and not weak_free,
                "lex_initial_known": initial_known,
                "lex_initial_stable": initial_stable,
                "lex_initial_blocked": initial_known and not initial_stable,
                "lex_final_known": lex_known,
                "lex_final_donor_free": lex_known and best_introduced == 0,
                "lex_final_assisted": lex_known and best_introduced > 0,
                "lex_best_introduced": best_introduced,
                "lex_best_used": best_used,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 1_440:
        raise ValueError(f"Expected 1,440 market outcomes, found {len(result)}")
    return result


def validate_canonical_results(
    path: str | Path = CANONICAL_PATH,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    metadata, calls = load_canonical_results(path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Wrong canonical schema version")
    if calls["canonical_key"].duplicated().any():
        raise ValueError("Duplicate canonical result keys")
    primary = calls[calls["analysis_role"].eq("primary_full_core")]
    if primary.empty or not primary["full_core"].eq(True).all():
        raise ValueError("Every primary result row must be marked as full-core evidence")
    initial = primary[
        primary["procedure"].eq("lexicographic_rule")
        & primary["stage"].eq("initial_audit")
    ]
    if len(initial) != 1_440 or initial["market_id"].nunique() != 1_440:
        raise ValueError("Initial lexicographic design is incomplete")
    if set(initial["instance"]) != {name for name, _ in FROZEN_INSTANCES}:
        raise ValueError("Initial design does not use the frozen 30 instances")
    cell_counts = initial.groupby(
        ["pool_size", "num_players", "partition_rep", "Delta"]
    ).size()
    if not cell_counts.eq(30).all() or len(cell_counts) != 48:
        raise ValueError("The final design must contain 30 base markets in every partitioned cell")
    coarse_counts = initial.groupby(["pool_size", "num_players", "Delta"]).size()
    if not coarse_counts.eq(60).all() or len(coarse_counts) != 24:
        raise ValueError("The final design must contain 60 markets in every cell")
    outcomes = market_outcomes(path)
    if not outcomes["weak_known"].all():
        raise ValueError("At least one market lacks a certified weak-core witness")
    blocked_ids = set(outcomes.loc[outcomes["lex_initial_blocked"], "market_id"])
    floor = primary[
        primary["procedure"].eq("lexicographic_rule")
        & primary["stage"].eq("floor_stabilization")
    ]
    if set(floor["market_id"]) != blocked_ids:
        raise ValueError("Floor-search scope differs from certified initial blocks")
    for market_id, group in floor.groupby("market_id"):
        order0 = group[group["order_rep"].eq(0)]
        zero_success = bool(
            len(order0) == 1
            and order0.iloc[0].get("certified")
            and order0.iloc[0].get("in_core")
            and int(order0.iloc[0].get("donors_introduced") or 0) == 0
        )
        if not zero_success and set(group["order_rep"].astype(int)) != {0, 1, 2}:
            raise ValueError(f"Incomplete conditional donor-order study for {market_id}")

    successful_primary = primary[
        primary["certified"].eq(True) & primary["in_core"].eq(True)
    ]
    for _, row in successful_primary.iterrows():
        introduced = row.get("introduced_donor_ids") or []
        used = row.get("used_donor_ids") or []
        if len(introduced) != int(row.get("donors_introduced") or 0):
            raise ValueError(f"Introduced-donor mismatch in {row['canonical_key']}")
        if len(used) != int(row.get("donors_used") or 0):
            raise ValueError(f"Used-donor mismatch in {row['canonical_key']}")
        if not set(used).issubset(introduced):
            raise ValueError(f"Unintroduced donor used in {row['canonical_key']}")
        if row.get("procedure") != "lexicographic_rule" or row.get("stage") != "floor_stabilization":
            continue
        floors = row.get("baseline_objective_tiers") or {}
        achieved = row.get("achieved_objective_tiers") or row.get("objective_tiers") or {}
        for name in ("transplants", "cycle_count", "same_blood"):
            if name in floors and abs(float(floors[name]) - round(float(floors[name]))) > 1e-6:
                raise ValueError(f"Nonintegral stored {name} floor in {row['canonical_key']}")
        for name, value in floors.items():
            tolerance = 1e-6 if name != "hard_match" else 1e-7
            if float(achieved[name]) + tolerance < float(value):
                raise ValueError(f"Objective floor violation in {row['canonical_key']}")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "canonical_file": str(Path(path)),
        "canonical_sha256": sha256_file(path),
        "result_call_rows": len(calls),
        "markets": len(outcomes),
        "weak_certified_markets": int(outcomes["weak_known"].sum()),
        "weak_donor_free_markets": int(outcomes["weak_donor_free"].sum()),
        "weak_assisted_markets": int(outcomes["weak_assisted"].sum()),
        "lex_initial_stable": int(outcomes["lex_initial_stable"].sum()),
        "lex_initial_blocked": int(outcomes["lex_initial_blocked"].sum()),
        "lex_initial_unresolved": int((~outcomes["lex_initial_known"]).sum()),
        "lex_final_donor_free": int(outcomes["lex_final_donor_free"].sum()),
        "lex_final_assisted": int(outcomes["lex_final_assisted"].sum()),
        "lex_final_unresolved": int((~outcomes["lex_final_known"]).sum()),
    }
    if require_complete and (
        summary["lex_initial_unresolved"]
        or summary["lex_final_unresolved"]
        or summary["weak_certified_markets"] != summary["markets"]
    ):
        raise ValueError("The primary full-core study still contains unresolved markets")
    return summary


def _cluster_bootstrap_rate(
    frame: pd.DataFrame,
    value: str,
    known: str,
    *,
    seed: int,
) -> tuple[float, float, float]:
    sample = frame[frame[known]].copy()
    clusters = sample.groupby("instance")[value].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        clusters, size=(BOOTSTRAP_REPS, len(clusters)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(clusters.mean()), float(low), float(high)


def _cluster_bootstrap_pool_correlation(
    frame: pd.DataFrame,
    value: str,
    *,
    seed: int,
) -> tuple[float, float, float]:
    """Spearman correlation with a base-instance cluster bootstrap."""
    ranked = frame.assign(
        _pool_rank=frame["pool_size"].map({100: 1.0, 200: 2.0, 500: 3.0}),
        _outcome=frame[value].astype(float),
    )
    aggregates = []
    for _, group in ranked.groupby("instance", sort=True):
        x = group["_pool_rank"].to_numpy(float)
        y = group["_outcome"].to_numpy(float)
        aggregates.append(
            [
                len(group),
                x.sum(),
                y.sum(),
                np.square(x).sum(),
                np.square(y).sum(),
                np.multiply(x, y).sum(),
            ]
        )
    stats = np.asarray(aggregates, dtype=float)

    def correlation(summed: np.ndarray) -> np.ndarray:
        n, sx, sy, sxx, syy, sxy = np.moveaxis(summed, -1, 0)
        numerator = n * sxy - sx * sy
        denominator = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 0,
        )

    estimate = float(correlation(stats.sum(axis=0)))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(stats), size=(BOOTSTRAP_REPS, len(stats)))
    draws = correlation(stats[indices].sum(axis=1))
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def analysis_tables(path: str | Path = CANONICAL_PATH) -> dict[str, pd.DataFrame]:
    _, calls = load_canonical_results(path)
    markets = market_outcomes(path)
    outcome_rows = []
    for n, group in markets.groupby("num_players", sort=True):
        outcome_rows.append(
            {
                "num_players": int(n),
                "weak_donor_free": int(group["weak_donor_free"].sum()),
                "weak_assisted": int(group["weak_assisted"].sum()),
                "weak_unresolved": int((~group["weak_known"]).sum()),
                "lex_initial_stable": int(group["lex_initial_stable"].sum()),
                "lex_initial_blocked": int(group["lex_initial_blocked"].sum()),
                "lex_initial_unresolved": int((~group["lex_initial_known"]).sum()),
                "lex_final_donor_free": int(group["lex_final_donor_free"].sum()),
                "lex_final_assisted": int(group["lex_final_assisted"].sum()),
                "lex_final_unresolved": int((~group["lex_final_known"]).sum()),
            }
        )
    outcomes = pd.DataFrame(outcome_rows)

    parameter_rows: list[dict[str, Any]] = []
    parameter_groups: list[tuple[str, str, pd.DataFrame]] = [
        ("Overall", "All", markets)
    ]
    parameter_groups.extend(
        ("Organizations", str(level), group)
        for level, group in markets.groupby("num_players", sort=True)
    )
    parameter_groups.extend(
        ("Pool size", str(level), group)
        for level, group in markets.groupby("pool_size", sort=True)
    )
    parameter_groups.extend(
        ("Cycle cap", str(level), group)
        for level, group in markets.groupby("Delta", sort=True)
    )
    for index, (factor, level, group) in enumerate(parameter_groups):
        initial_rate, initial_low, initial_high = _cluster_bootstrap_rate(
            group,
            "lex_initial_blocked",
            "lex_initial_known",
            seed=BOOTSTRAP_SEED + 3 * index,
        )
        assisted_rate, assisted_low, assisted_high = _cluster_bootstrap_rate(
            group,
            "lex_final_assisted",
            "lex_final_known",
            seed=BOOTSTRAP_SEED + 3 * index + 1,
        )
        weak_rate, weak_low, weak_high = _cluster_bootstrap_rate(
            group,
            "weak_assisted",
            "weak_known",
            seed=BOOTSTRAP_SEED + 3 * index + 2,
        )
        parameter_rows.append(
            {
                "factor": factor,
                "level": level,
                "markets": len(group),
                "initial_blocked": int(group["lex_initial_blocked"].sum()),
                "initial_block_rate": initial_rate,
                "initial_block_ci_low": initial_low,
                "initial_block_ci_high": initial_high,
                "lex_assisted": int(group["lex_final_assisted"].sum()),
                "lex_assisted_rate": assisted_rate,
                "lex_assisted_ci_low": assisted_low,
                "lex_assisted_ci_high": assisted_high,
                "weak_assisted": int(group["weak_assisted"].sum()),
                "weak_assisted_rate": weak_rate,
                "weak_assisted_ci_low": weak_low,
                "weak_assisted_ci_high": weak_high,
            }
        )
    parameter_summary = pd.DataFrame(parameter_rows)

    cell_effects = (
        markets.groupby(["pool_size", "num_players", "Delta"], as_index=False)
        .agg(
            markets=("market_id", "size"),
            initial_blocked=("lex_initial_blocked", "sum"),
            lex_assisted=("lex_final_assisted", "sum"),
            weak_assisted=("weak_assisted", "sum"),
        )
        .sort_values(["pool_size", "num_players", "Delta"])
    )
    cell_effects["initial_block_rate"] = (
        cell_effects["initial_blocked"] / cell_effects["markets"]
    )
    cell_effects["lex_assisted_rate"] = (
        cell_effects["lex_assisted"] / cell_effects["markets"]
    )

    pool_rows = []
    for pool, group in markets.groupby("pool_size", sort=True):
        initial_known = group[group["lex_initial_known"]]
        final_known = group[group["lex_final_known"]]
        challenged_known = group[
            group["lex_initial_blocked"] & group["lex_final_known"]
        ]
        assisted_known = challenged_known[challenged_known["lex_final_assisted"]]
        pool_rows.append(
            {
                "pool_size": int(pool),
                "markets": len(group),
                "certified_initial": len(initial_known),
                "initial_blocked": int(initial_known["lex_initial_blocked"].sum()),
                "initial_block_rate": initial_known["lex_initial_blocked"].mean(),
                "final_resolved": len(final_known),
                "donor_assisted": int(final_known["lex_final_assisted"].sum()),
                "donor_assisted_rate": final_known["lex_final_assisted"].mean(),
                "successful_challenged": len(challenged_known),
                "challenged_best_prefix_mean": challenged_known[
                    "lex_best_introduced"
                ].mean(),
                "challenged_best_prefix_median": challenged_known[
                    "lex_best_introduced"
                ].median(),
                "challenged_best_prefix_max": challenged_known[
                    "lex_best_introduced"
                ].max(),
                "assisted_best_prefix_mean": assisted_known[
                    "lex_best_introduced"
                ].mean(),
                "assisted_best_prefix_median": assisted_known[
                    "lex_best_introduced"
                ].median(),
                "assisted_best_prefix_max": assisted_known[
                    "lex_best_introduced"
                ].max(),
            }
        )
    pools = pd.DataFrame(pool_rows)

    floor = calls[
        calls["procedure"].eq("lexicographic_rule")
        & calls["stage"].eq("floor_stabilization")
    ]
    donor_rows = []
    for market_id, group in floor.groupby("market_id"):
        success = group[group["certified"].eq(True) & group["in_core"].eq(True)]
        if success.empty:
            continue
        first = success.iloc[0]
        donor_rows.append(
            {
                "market_id": market_id,
                "num_players": int(first["num_players"]),
                "pool_size": int(first["pool_size"]),
                "Delta": int(first["Delta"]),
                "successful_orders": success["order_rep"].nunique(),
                "best_introduced": int(success["donors_introduced"].min()),
                "worst_introduced": int(success["donors_introduced"].max()),
                "best_used": int(success["donors_used"].min()),
                "worst_used": int(success["donors_used"].max()),
            }
        )
    lex_donors = pd.DataFrame(donor_rows)
    assisted_donors = lex_donors[lex_donors["best_introduced"].gt(0)]
    order_sensitivity = pd.DataFrame(
        [
            {
                "initially_blocked_markets": len(lex_donors),
                "alternate_donor_free_markets": int(
                    lex_donors["best_introduced"].eq(0).sum()
                ),
                "assisted_markets": len(assisted_donors),
                "best_introduced_mean": assisted_donors["best_introduced"].mean(),
                "best_introduced_median": assisted_donors[
                    "best_introduced"
                ].median(),
                "best_introduced_max": assisted_donors["best_introduced"].max(),
                "best_used_mean": assisted_donors["best_used"].mean(),
                "best_used_max": assisted_donors["best_used"].max(),
                "introduced_varies_across_orders": int(
                    assisted_donors["best_introduced"].ne(
                        assisted_donors["worst_introduced"]
                    ).sum()
                ),
                "used_varies_across_orders": int(
                    assisted_donors["best_used"].ne(
                        assisted_donors["worst_used"]
                    ).sum()
                ),
                "three_certified_orders": int(
                    assisted_donors["successful_orders"].eq(3).sum()
                ),
                "two_certified_orders": int(
                    assisted_donors["successful_orders"].eq(2).sum()
                ),
            }
        ]
    )

    base_instances = (
        markets.groupby("instance", as_index=False)
        .agg(
            markets=("market_id", "size"),
            initial_blocked=("lex_initial_blocked", "sum"),
            lex_assisted=("lex_final_assisted", "sum"),
            weak_assisted=("weak_assisted", "sum"),
        )
        .sort_values("instance")
    )
    for column in ("initial_blocked", "lex_assisted", "weak_assisted"):
        base_instances[f"{column}_rate"] = base_instances[column] / base_instances["markets"]

    paired = markets.pivot_table(
        index=["instance", "pool_size", "num_players", "Delta"],
        columns="partition_rep",
        values=["lex_initial_blocked", "lex_final_assisted", "weak_assisted"],
        aggfunc="first",
    )
    partition_discordance = pd.DataFrame(
        [
            {
                "outcome": label,
                "paired_settings": len(paired),
                "discordant_settings": int(paired[column][0].ne(paired[column][1]).sum()),
            }
            for column, label in (
                ("lex_initial_blocked", "Initial lexicographic block"),
                ("lex_final_assisted", "Positive lexicographic assistance"),
                ("weak_assisted", "Positive weak-core assistance"),
            )
        ]
    )
    partition_discordance["discordant_rate"] = (
        partition_discordance["discordant_settings"]
        / partition_discordance["paired_settings"]
    )

    pool_rho, pool_rho_low, pool_rho_high = _cluster_bootstrap_pool_correlation(
        markets,
        "lex_final_assisted",
        seed=BOOTSTRAP_SEED + 10_000,
    )
    pool_correlation = pd.DataFrame(
        [
            {
                "outcome": "Positive lexicographic assistance",
                "spearman_rho": pool_rho,
                "cluster_ci_low": pool_rho_low,
                "cluster_ci_high": pool_rho_high,
            }
        ]
    )

    supplementary_rows = []
    supplement = calls[calls["analysis_role"].eq("supplementary_robustness")]
    for procedure in ("tu_core", "strong_core"):
        procedure_rows = supplement[supplement["procedure"].eq(procedure)]
        for n in PLAYERS:
            group = procedure_rows[procedure_rows["num_players"].eq(n)]
            market_ids = set(group["market_id"])
            stable = group[group["certified"].eq(True) & group["in_core"].eq(True)]
            free = set(stable.loc[stable["donors_introduced"].eq(0), "market_id"])
            assisted = set(stable["market_id"]) - free
            best = (
                stable[stable["market_id"].isin(assisted)]
                .groupby("market_id")["donors_introduced"]
                .min()
            )
            supplementary_rows.append(
                {
                    "procedure": procedure,
                    "num_players": n,
                    "donor_free": len(free),
                    "assisted": len(assisted),
                    "unresolved": 360 - len(free) - len(assisted),
                    "best_introduced_mean": best.mean(),
                    "best_introduced_median": best.median(),
                    "best_introduced_max": best.max(),
                    "market_rows_found": len(market_ids),
                }
            )
    supplementary = pd.DataFrame(supplementary_rows)

    actual_weak = calls[
        calls["procedure"].eq("weak_core")
        & calls["source_algorithm"].eq("weak_heuristic")
    ]
    initial_calls = calls[
        calls["procedure"].eq("lexicographic_rule")
        & calls["stage"].eq("initial_audit")
    ]
    runtime_groups = {
        "Weak heuristic": actual_weak,
        "Initial lexicographic audit": initial_calls,
        "Floor-preserving stabilization": floor,
    }
    runtime_rows = []
    for label, group in runtime_groups.items():
        seconds = pd.to_numeric(group["runner_seconds"], errors="coerce")
        runtime_rows.append(
            {
                "procedure": label,
                "calls": len(group),
                "certified_calls": int(group["certified"].eq(True).sum()),
                "median_seconds": seconds.median(),
                "p90_seconds": seconds.quantile(0.9),
                "p95_seconds": seconds.quantile(0.95),
                "max_seconds": seconds.max(),
                "elapsed_hours": seconds.sum() / 3600,
            }
        )
    runtimes = pd.DataFrame(runtime_rows)
    return {
        "markets": markets,
        "outcomes": outcomes,
        "parameter_summary": parameter_summary,
        "cell_effects": cell_effects,
        "pools": pools,
        "lex_donors": lex_donors,
        "order_sensitivity": order_sensitivity,
        "base_instances": base_instances,
        "partition_discordance": partition_discordance,
        "pool_correlation": pool_correlation,
        "supplementary": supplementary,
        "runtimes": runtimes,
    }


def _heatmap(
    ax: plt.Axes,
    markets: pd.DataFrame,
    *,
    delta: int,
    value: str,
    known: str,
    title: str,
    vmax: float,
) -> Any:
    matrix = np.zeros((len(POOLS), len(PLAYERS)))
    denominators = np.zeros_like(matrix, dtype=int)
    for row_index, pool in enumerate(POOLS):
        for column_index, n in enumerate(PLAYERS):
            cell = markets[
                markets["pool_size"].eq(pool)
                & markets["num_players"].eq(n)
                & markets["Delta"].eq(delta)
            ]
            sample = cell[cell[known]]
            matrix[row_index, column_index] = 100 * sample[value].mean()
            denominators[row_index, column_index] = len(sample)
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            label = f"{matrix[row_index, column_index]:.1f}"
            if denominators[row_index, column_index] < 60:
                label += f"\nN={denominators[row_index, column_index]}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row_index, column_index] > 0.55 * vmax else "black",
            )
    ax.set_xticks(range(len(PLAYERS)), PLAYERS)
    ax.set_yticks(range(len(POOLS)), POOLS)
    ax.set_xlabel("Number of organizations")
    ax.set_ylabel("Pool size")
    ax.set_title(title)
    return image


def build_figures(
    path: str | Path = CANONICAL_PATH,
    figure_dir: str | Path = FIGURE_DIR,
) -> list[Path]:
    tables = analysis_tables(path)
    markets = tables["markets"]
    pools = tables["pools"]
    donors = tables["lex_donors"]
    runtimes = tables["runtimes"]
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    outputs: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.25), constrained_layout=True)
    images = []
    for column, delta in enumerate(DELTAS):
        images.append(
            _heatmap(
                axes[0, column],
                markets,
                delta=delta,
                value="lex_initial_blocked",
                known="lex_initial_known",
                title=rf"Initial allocation blocked, $\Delta={delta}$",
                vmax=35,
            )
        )
        images.append(
            _heatmap(
                axes[1, column],
                markets,
                delta=delta,
                value="lex_final_assisted",
                known="lex_final_known",
                title=rf"Positive donor assistance, $\Delta={delta}$",
                vmax=30,
            )
        )
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Share of resolved markets (%)")
    output = figure_dir / "full_core_parameter_effects.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), constrained_layout=True)
    x = np.arange(len(POOLS))
    axes[0].plot(x, 100 * pools["initial_block_rate"], marker="o", label="Initial blocked")
    axes[0].plot(x, 100 * pools["donor_assisted_rate"], marker="s", label="Positive assistance")
    axes[0].set_xticks(x, POOLS)
    axes[0].set_xlabel("Pool size")
    axes[0].set_ylabel("Share of resolved markets (%)")
    axes[0].set_title("A. Frequency of instability and assistance")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].legend(frameon=False)

    successful = donors[donors["best_introduced"].gt(0)]
    series = [
        successful[successful["pool_size"].eq(pool)]["best_introduced"].to_numpy()
        for pool in POOLS
    ]
    axes[1].boxplot(series, tick_labels=POOLS, showfliers=True, widths=0.55)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for index, values in enumerate(series, start=1):
        axes[1].scatter(
            rng.normal(index, 0.035, len(values)),
            values,
            s=12,
            alpha=0.45,
            color="#D55E00",
            linewidths=0,
        )
    axes[1].set_xlabel("Pool size")
    axes[1].set_ylabel("Best-tested donors introduced")
    axes[1].set_title("B. Markets with positive donor assistance")
    axes[1].set_ylim(0.8, 2.2)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    output = figure_dir / "pool_size_and_donor_assistance.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    fig, ax = plt.subplots(figsize=(7.35, 3.0), constrained_layout=True)
    x = np.arange(len(runtimes))
    bars = ax.bar(x, runtimes["median_seconds"], color=["#009E73", "#0072B2", "#D55E00"], alpha=0.75)
    ax.scatter(x, runtimes["p90_seconds"], marker="D", color="black", s=28, label="90th percentile")
    ax.set_yscale("log")
    ax.set_xticks(x, ["Weak heuristic", "Initial lex audit", "Floor stabilization"])
    ax.set_ylabel("Elapsed seconds per call (log scale)")
    ax.grid(axis="y", which="both", color="#E0E0E0", linewidth=0.5)
    ax.legend(frameon=False)
    for bar, calls in zip(bars, runtimes["calls"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15, f"N={calls}", ha="center", va="bottom", fontsize=8)
    output = figure_dir / "full_core_runtime_profiles.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)
    return outputs


def write_analysis_tables(
    path: str | Path = CANONICAL_PATH,
    table_dir: str | Path = TABLE_DIR,
) -> list[Path]:
    """Persist all derived analysis tables without creating another raw dataset."""
    tables = analysis_tables(path)
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "markets": "market_outcomes.csv",
        "outcomes": "primary_outcomes_by_organizations.csv",
        "parameter_summary": "parameter_summary_clustered_ci.csv",
        "cell_effects": "lexicographic_cell_effects.csv",
        "pools": "pool_size_summary.csv",
        "lex_donors": "lexicographic_donor_orders.csv",
        "order_sensitivity": "lexicographic_order_sensitivity.csv",
        "base_instances": "base_instance_summary.csv",
        "partition_discordance": "partition_discordance.csv",
        "pool_correlation": "pool_size_correlation.csv",
        "supplementary": "supplementary_tu_strong.csv",
        "runtimes": "runtime_summary.csv",
    }
    outputs: list[Path] = []
    for name, filename in filenames.items():
        output = table_dir / filename
        tables[name].to_csv(output, index=False)
        outputs.append(output)
    manifest = table_dir / "analysis_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "canonical_file": Path(path).as_posix(),
                "canonical_sha256": sha256_file(path),
                "bootstrap_repetitions": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "tables": [output.name for output in outputs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(manifest)
    return outputs


def _retry_checkpoint_rows(path: Path = RETRY_PATH) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = (
            str(row["market_id"]),
            str(row["stage"]),
            int(row.get("order_rep", 0)),
        )
        rows[key] = row
    return rows


def _append_retry(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(_clean_json_value(dict(row)), sort_keys=True, allow_nan=False)
            + "\n"
        )


def _baseline_as_result(
    baseline: Mapping[str, Any], donor_order: Sequence[int]
) -> dict[str, Any]:
    certified = bool(baseline.get("certified"))
    in_core = bool(baseline.get("in_core"))
    if certified and in_core:
        reason = "stable_donor_free_lexicographic_allocation"
    elif certified:
        reason = "blocked_donor_free_lexicographic_allocation"
    else:
        reason = "donor_free_lexicographic_audit_not_certified"
    return {
        "solution": set(baseline.get("selection", set())),
        "final_in_core": in_core,
        "in_core": in_core,
        "certified": certified,
        "termination_reason": reason,
        "solver_status": baseline.get("solver_status"),
        "objective_tiers": dict(baseline.get("objective_tiers", {})),
        "altruists_introduced": 0,
        "altruists_added": 0,
        "altruists_used": 0,
        "introduced_donor_ids": [],
        "used_donor_ids": [],
        "donor_order": list(donor_order),
        "runtime_seconds": float(baseline.get("runtime_seconds", 0.0)),
        "cycle_enumeration_seconds": baseline.get("cycle_enumeration_seconds"),
        "optimization_seconds": baseline.get("optimization_seconds"),
        "separation_seconds": baseline.get("separation_seconds"),
        "num_optimization_solves": baseline.get("num_optimization_solves"),
        "num_separation_solves": baseline.get("num_separation_solves"),
        "num_cuts": 0,
        "max_mip_gap": None,
    }


def _retry_base_fields(
    row: Mapping[str, Any], *, market_id: str | None = None
) -> dict[str, Any]:
    keys = (
        "market_id",
        "instance",
        "pool_size",
        "num_players",
        "partition_rep",
        "Delta",
        "actual_vertices",
        "available_altruists",
        "graph_seed",
        "partition_seed",
        "graph_construction_seconds",
        "partition_seconds",
        "instance_load_seconds",
    )
    fields = {key: row.get(key) for key in keys if key in row}
    if market_id is not None:
        fields["market_id"] = market_id
    return fields


def _retry_record(
    row: Mapping[str, Any],
    *,
    stage: str,
    cap: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    record = {str(key): _clean_json_value(value) for key, value in row.items()}
    record.update(
        {
            "record_type": "result_call",
            "schema_version": SCHEMA_VERSION,
            "analysis_role": "primary_full_core",
            "procedure": "lexicographic_rule",
            "stage": stage,
            "source_algorithm": str(record.get("algorithm")),
            "source_file": checkpoint_path.as_posix(),
            "source_file_sha256": None,
            "effective_coalition_cap": cap,
            "max_coal_size": cap,
            "full_core": True,
            "full_core_basis": _full_core_basis(
                int(record["num_players"]),
                min(cap, int(record["num_players"])),
            ),
        }
    )
    if stage == "floor_stabilization":
        record.setdefault(
            "objective_floor_type", "donor_free_lexicographic_lower_bounds"
        )
        record.setdefault("search_cut_type", "aggregate_weak_core_heuristic")
    record["canonical_key"] = "|".join(
        [
            "primary_full_core",
            "lexicographic_rule",
            stage,
            str(record["market_id"]),
            str(record["source_algorithm"]),
            str(int(record.get("order_rep") or 0)),
        ]
    )
    return record


def inconclusive_lexicographic_targets(
    calls: pd.DataFrame,
) -> dict[str, str]:
    """Return the lexicographic markets that need the long-limit pass.

    Initial audits take priority.  A certified blocked initial allocation is
    a floor-search target only when none of its stored donor-order searches is
    already a certified stable result.  Sorting by market identifier makes the
    checkpoint order reproducible.
    """
    primary = calls[
        calls["analysis_role"].eq("primary_full_core")
        & calls["procedure"].eq("lexicographic_rule")
    ]
    initial = primary[primary["stage"].eq("initial_audit")]
    floor = primary[primary["stage"].eq("floor_stabilization")]
    targets: dict[str, str] = {}
    for _, row in initial.sort_values("market_id").iterrows():
        market_id = str(row["market_id"])
        certified = bool(pd.notna(row.get("certified")) and row.get("certified"))
        stable = bool(pd.notna(row.get("in_core")) and row.get("in_core"))
        if not certified:
            targets[market_id] = "initial_audit"
            continue
        if stable:
            continue
        market_floor = floor[floor["market_id"].eq(market_id)]
        has_stable_floor = bool(
            (
                market_floor["certified"].eq(True)
                & market_floor["in_core"].eq(True)
            ).any()
        )
        if not has_stable_floor:
            targets[market_id] = "floor_stabilization"
    return targets


def retry_inconclusive_lexicographic_markets(
    canonical_path: str | Path = CANONICAL_PATH,
    checkpoint_path: str | Path = RETRY_PATH,
    *,
    time_limit_seconds: int = 1_200,
    solver_threads: int = 8,
) -> dict[str, Any]:
    """Selectively rerun every unresolved primary lexicographic market.

    The exact frozen instances and all scientific seeds are reused.  Only the
    per-MIP time limit and thread count differ.  Results are checkpointed after
    every call; :func:`merge_selective_retries` promotes conditionally complete
    retries into the canonical one-file dataset.
    """
    from instance_analysis import load_instance
    from KEP_functions import (
        build_compat_graph,
        build_graph_features,
        lexicographic_floor_core_search,
        make_partition,
        prepare_lexicographic_floor_baseline,
    )
    canonical_path = Path(canonical_path)
    checkpoint_path = Path(checkpoint_path)
    _, calls = load_canonical_results(canonical_path)
    initial = calls[
        calls["procedure"].eq("lexicographic_rule")
        & calls["stage"].eq("initial_audit")
    ].set_index("market_id")
    targets = inconclusive_lexicographic_targets(calls)
    completed = _retry_checkpoint_rows(checkpoint_path)
    started = perf_counter()
    new_rows = 0

    for target_index, (market_id, target_stage) in enumerate(
        targets.items(), start=1
    ):
        initial_retry = completed.get((market_id, "initial_audit", -1))
        floor_retries = {
            order: completed.get((market_id, "floor_stabilization", order))
            for order in range(3)
        }
        order0_retry = floor_retries[0]
        floor_complete = bool(
            order0_retry is not None
            and (
                (
                    bool(order0_retry.get("certified"))
                    and bool(order0_retry.get("in_core"))
                    and int(order0_retry.get("donors_introduced") or 0) == 0
                )
                or all(floor_retries[order] is not None for order in range(3))
            )
        )
        target_complete = floor_complete
        if target_stage == "initial_audit" and initial_retry is not None:
            target_complete = bool(initial_retry.get("certified")) and (
                bool(initial_retry.get("in_core")) or floor_complete
            )
        if target_complete:
            print(
                f"[{target_index}/{len(targets)}] {market_id}: "
                "long-limit retry already complete"
            )
            continue

        source = initial.loc[market_id]
        instance_name = str(source["instance"])
        pool_size = int(source["pool_size"])
        num_players = int(source["num_players"])
        partition_rep = int(source["partition_rep"])
        delta = int(source["Delta"])
        cap = 9 if num_players <= 10 else 30

        instance_path = Path("instances_large") / instance_name
        expected_hash = dict(FROZEN_INSTANCES)[instance_name]
        if sha256_file(instance_path) != expected_hash:
            raise ValueError(f"Instance hash mismatch for {instance_path}")
        instance = load_instance(instance_path)
        features = build_graph_features(instance)
        graph_seed = stable_seed(MASTER_SEED, "graph", instance_name, pool_size)
        vertices, adj_out, _, altruist_edges = build_compat_graph(
            instance,
            num_patients=pool_size,
            rng=np.random.default_rng(graph_seed),
        )
        partition_seed = stable_seed(
            MASTER_SEED,
            "partition",
            instance_name,
            pool_size,
            num_players,
            partition_rep,
        )
        partition = make_partition(
            vertices,
            num_players=num_players,
            var_size=1,
            rng=np.random.default_rng(partition_seed),
        )
        donor_orders = make_donor_orders(
            altruist_edges,
            master_seed=MASTER_SEED,
            instance_name=instance_name,
            pool_size=pool_size,
            num_players=num_players,
            partition_rep=partition_rep,
            repetitions=3,
        )
        solver_seed = stable_seed(
            MASTER_SEED,
            "solver",
            instance_name,
            pool_size,
            num_players,
            partition_rep,
            delta,
        )

        baseline_started = perf_counter()
        baseline = prepare_lexicographic_floor_baseline(
            vertices,
            adj_out,
            partition,
            delta,
            features,
            max_coal_size=cap,
            solver="GUROBI",
            time_limit=time_limit_seconds,
            mip_gap=0.0,
            threads=solver_threads,
            solver_seed=stable_seed(solver_seed, "lex_initial_solver"),
        )
        baseline_elapsed = perf_counter() - baseline_started

        if target_stage == "initial_audit":
            key = (market_id, "initial_audit", -1)
            if key not in completed:
                flat = _flatten_result(
                    _retry_base_fields(source, market_id=market_id),
                    algorithm="lexicographic_initial_retry",
                    order_rep=-1,
                    evidence_type="long_limit_exact_full_weak_core_initial_audit",
                    result=_baseline_as_result(baseline, donor_orders[0]),
                    runner_seconds=baseline_elapsed,
                )
                retry = _retry_record(
                    flat,
                    stage="initial_audit",
                    cap=cap,
                    checkpoint_path=checkpoint_path,
                )
                _append_retry(checkpoint_path, retry)
                completed[key] = retry
                new_rows += 1
            if not bool(baseline.get("certified")) or bool(baseline.get("in_core")):
                print(
                    f"[{target_index}/{len(targets)}] {market_id}: "
                    f"initial certified={baseline.get('certified')} stable={baseline.get('in_core')}"
                )
                continue

        # A certified blocked baseline is required for the floor search.  For
        # an existing floor target, a newly found stable tied baseline is itself
        # a successful zero-donor floor-preserving result.
        orders = range(3)
        for order_rep in orders:
            key = (market_id, "floor_stabilization", order_rep)
            if key in completed:
                retry = completed[key]
            else:
                search_started = perf_counter()
                result = lexicographic_floor_core_search(
                    vertices,
                    adj_out,
                    partition,
                    delta,
                    features,
                    max_coal_size=cap,
                    solver="GUROBI",
                    altruist_edges=altruist_edges,
                    rng=np.random.default_rng(
                        stable_seed(solver_seed, "lex", order_rep)
                    ),
                    donor_order=donor_orders[order_rep],
                    time_limit=time_limit_seconds,
                    mip_gap=0.0,
                    threads=solver_threads,
                    solver_seed=stable_seed(
                        solver_seed, "lex_solver", order_rep
                    ),
                    baseline=baseline,
                )
                elapsed = perf_counter() - search_started
                flat = _flatten_result(
                    _retry_base_fields(source, market_id=market_id),
                    algorithm="lexicographic_floor_retry",
                    order_rep=order_rep,
                    evidence_type="long_limit_frozen_floor_full_weak_core_search",
                    result=result,
                    runner_seconds=elapsed + (baseline_elapsed if order_rep == 0 else 0.0),
                )
                flat.update(
                    {
                        "baseline_objective_tiers": dict(
                            result.get("baseline_objective_tiers", {})
                        ),
                        "achieved_objective_tiers": dict(
                            result.get("objective_tiers", {})
                        ),
                        "objective_floor_slacks": dict(
                            result.get("objective_floor_slacks", {})
                        ),
                    }
                )
                retry = _retry_record(
                    flat,
                    stage="floor_stabilization",
                    cap=cap,
                    checkpoint_path=checkpoint_path,
                )
                _append_retry(checkpoint_path, retry)
                completed[key] = retry
                new_rows += 1
            print(
                f"[{target_index}/{len(targets)}] {market_id} "
                f"order={order_rep}: certified={retry.get('certified')} "
                f"stable={retry.get('in_core')} introduced={retry.get('donors_introduced')}"
            )
            if (
                order_rep == 0
                and bool(retry.get("certified"))
                and bool(retry.get("in_core"))
                and int(retry.get("donors_introduced") or 0) == 0
            ):
                break

    merge = merge_selective_retries(
        canonical_path,
        checkpoint_path,
        time_limit_seconds=time_limit_seconds,
        solver_threads=solver_threads,
        targets=targets,
    )
    return {
        "targets": len(targets),
        "new_checkpoint_rows": new_rows,
        "elapsed_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        **merge,
    }


def merge_selective_retries(
    canonical_path: str | Path = CANONICAL_PATH,
    checkpoint_path: str | Path = RETRY_PATH,
    *,
    time_limit_seconds: int = 1_200,
    solver_threads: int = 8,
    targets: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Promote complete retry targets and rebuild the canonical JSONL."""
    canonical_path = Path(canonical_path)
    checkpoint_path = Path(checkpoint_path)
    metadata, calls = load_canonical_results(canonical_path)
    targets = dict(targets or inconclusive_lexicographic_targets(calls))
    retry = _retry_checkpoint_rows(checkpoint_path)
    promoted: set[str] = set()
    additions: list[dict[str, Any]] = []

    for market_id, target_stage in targets.items():
        initial_row = retry.get((market_id, "initial_audit", -1))
        floor_rows = {
            order: retry.get((market_id, "floor_stabilization", order))
            for order in range(3)
        }
        complete_floor = False
        order0 = floor_rows[0]
        if order0 is not None:
            zero_success = (
                bool(order0.get("certified"))
                and bool(order0.get("in_core"))
                and int(order0.get("donors_introduced") or 0) == 0
            )
            complete_floor = zero_success or all(
                floor_rows[order] is not None for order in range(3)
            )

        if target_stage == "initial_audit":
            if initial_row is None:
                continue
            if bool(initial_row.get("certified")) and not bool(initial_row.get("in_core")):
                if not complete_floor:
                    continue
            promoted.add(market_id)
            additions.append(initial_row)
            if complete_floor:
                additions.extend(row for row in floor_rows.values() if row is not None)
        elif complete_floor:
            promoted.add(market_id)
            additions.extend(row for row in floor_rows.values() if row is not None)

    if promoted:
        checkpoint_sha256 = sha256_file(checkpoint_path)
        normalized_additions = []
        for row in additions:
            normalized = dict(row)
            normalized["source_file"] = checkpoint_path.as_posix()
            normalized["source_file_sha256"] = checkpoint_sha256
            normalized["max_coal_size"] = int(
                normalized.get("effective_coalition_cap")
                or normalized.get("max_coal_size")
            )
            if normalized.get("stage") == "floor_stabilization":
                normalized.setdefault(
                    "objective_floor_type",
                    "donor_free_lexicographic_lower_bounds",
                )
                normalized.setdefault(
                    "search_cut_type", "aggregate_weak_core_heuristic"
                )
            normalized_additions.append(normalized)
        additions = normalized_additions
        remove = calls["market_id"].isin(promoted) & calls["procedure"].eq(
            "lexicographic_rule"
        ) & calls["stage"].isin(["initial_audit", "floor_stabilization"])
        # Existing certified initial rows remain authoritative for floor-only
        # retry targets; only their floor rows are replaced.
        floor_only = {
            market_id
            for market_id in promoted
            if targets[market_id] == "floor_stabilization"
        }
        keep_initial = (
            calls["market_id"].isin(floor_only)
            & calls["stage"].eq("initial_audit")
        )
        calls = calls[~remove | keep_initial]
        calls = pd.concat([calls, pd.DataFrame(additions)], ignore_index=True)

    metadata = dict(metadata)
    metadata["selective_retry"] = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "promoted_markets": sorted(promoted),
        "time_limit_seconds": int(time_limit_seconds),
        "solver_threads": int(solver_threads),
        "current_backend_sha256": sha256_file("KEP_functions.py"),
        "current_full_core_driver_sha256": sha256_file("paper_simulations.py"),
    }
    with canonical_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_clean_json_value(metadata), sort_keys=True, allow_nan=False) + "\n")
        for row in calls.to_dict("records"):
            handle.write(json.dumps(_clean_json_value(row), sort_keys=True, allow_nan=False) + "\n")
    validation = validate_canonical_results(canonical_path)
    return {
        "promoted_retry_markets": len(promoted),
        **validation,
    }


def build_all_from_completed_results(
    output_path: str | Path = CANONICAL_PATH,
    *,
    rebuild_canonical: bool = False,
    table_dir: str | Path = TABLE_DIR,
    figure_dir: str | Path = FIGURE_DIR,
) -> dict[str, Any]:
    if rebuild_canonical:
        validation = consolidate_completed_results(output_path)
    else:
        validation = validate_canonical_results(output_path, require_complete=True)
    tables = write_analysis_tables(output_path, table_dir)
    figures = build_figures(output_path, figure_dir)
    return {
        **validation,
        "tables": [str(path) for path in tables],
        "figures": [str(path) for path in figures],
    }


def paper_study_configs(
    work_dir: str | Path = "results/management_science_full_core/reproduction_work",
) -> tuple[StudyConfig, StudyConfig]:
    """Build the two resumable configurations used by the paper notebook."""
    work_dir = Path(work_dir)
    common = dict(
        instance_dir="instances_large",
        num_base_instances=30,
        instance_selection_seed=MASTER_SEED,
        master_seed=MASTER_SEED,
        pool_sizes=POOLS,
        deltas=DELTAS,
        partition_reps=2,
        partition_var_size=1,
        donor_order_reps=3,
        solver="GUROBI",
        mip_gap=0.0,
        run_heuristic_diagnostics=True,
        run_legacy_lexicographic_donor_search=False,
    )
    cap9 = StudyConfig(
        **common,
        output_dir=str(work_dir / "cap09"),
        num_players=PLAYERS,
        primary_num_players=(5, 10),
        max_coal_size=9,
        time_limit_seconds=200,
        solver_threads=16,
    )
    cap30 = robustness_config(work_dir / "cap30")
    return cap9, cap30


def run_complete_paper_study(
    *,
    run_optimizations: bool = False,
    work_dir: str | Path = "results/management_science_full_core/reproduction_work",
    canonical_path: str | Path = CANONICAL_PATH,
    retry_time_limit_seconds: int = 1_200,
    retry_solver_threads: int = 8,
) -> dict[str, Any]:
    """Run or validate the complete paper experiment through one entry point.

    The cap-nine checkpoint contains TU and strong-core robustness calls for
    all organization counts and the exhaustive weak/lexicographic calls for
    five and ten organizations.  The cap-30 checkpoint contains the full-core
    weak and lexicographic calls for 20 and 30 organizations.  Only unresolved
    lexicographic markets enter the final long-limit pass.  Every stage is
    append-only and safely resumes its own checkpoint.
    """
    work_dir = Path(work_dir)
    canonical_path = Path(canonical_path)
    retry_path = work_dir / "retry_checkpoint.jsonl"
    validate_frozen_instances()

    stages: dict[str, Any] = {}
    if run_optimizations:
        cap9, cap30 = paper_study_configs(work_dir)
        stages["cap9_tu_and_strong"] = run_study(
            cap9,
            active_num_players=PLAYERS,
            primary_num_players=(),
        )
        stages["low_n_weak_and_lexicographic_full_core"] = run_study(
            cap9,
            active_num_players=(5, 10),
            primary_num_players=(5, 10),
        )
        stages["cap9_lexicographic_floors"] = append_lexicographic_floor_results(
            cap9.output_dir
        )
        stages["cap30_weak_full_core"] = run_weak_full_core_robustness(cap30)
        stages["cap30_lexicographic_full_core"] = (
            run_lexicographic_full_core_robustness(cap30)
        )
        stages["consolidation"] = consolidate_completed_results(
            canonical_path,
            source_cap9=Path(cap9.output_dir) / RAW_NAME,
            source_cap30=Path(cap30.output_dir) / RAW_NAME,
        )
        stages["long_limit_retries"] = retry_inconclusive_lexicographic_markets(
            canonical_path,
            checkpoint_path=retry_path,
            time_limit_seconds=retry_time_limit_seconds,
            solver_threads=retry_solver_threads,
        )

    final = build_all_from_completed_results(
        canonical_path,
        rebuild_canonical=False,
    )
    return {
        "run_optimizations": bool(run_optimizations),
        "work_dir": str(work_dir),
        "canonical_path": str(canonical_path),
        "stages": stages,
        "final": final,
    }
