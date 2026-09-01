"""Full-core robustness study for n=20 and n=30 only.

The study reuses the exact base-instance selection, graph, partition, donor-
order, algorithm, and solver seeds of the cap-4/cap-9 experiments.  It exposes
two separately resumable entry points so the fast weak-core heuristic can be
completed before the potentially expensive lexicographic audit.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, Mapping, Tuple

import numpy as np
import pandas as pd

from instance_analysis import load_instance
from KEP_functions import (
    build_compat_graph,
    build_graph_features,
    core_heuristic,
    lexicographic_floor_core_search,
    make_partition,
    prepare_lexicographic_floor_baseline,
)
from management_science_simulations import (
    StudyConfig,
    _append_checkpoint,
    _flatten_result,
    _load_checkpoint,
    _market_base,
    _result_key,
    _run_call,
    make_donor_orders,
    select_base_instances,
    stable_seed,
    write_study_manifest,
)


OUTPUT_DIR = Path("results/management_science_maxcoal30_robustness")
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
    reference_path = Path(
        "results/management_science_maxcoal09_with_heuristics/study_manifest.json"
    )
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
        "seed_reference": "management_science_maxcoal09_with_heuristics",
        "code_hashes": {
            path.as_posix(): _sha256_file(path)
            for path in (
                Path("KEP_functions.py"),
                Path("management_science_simulations.py"),
                Path("management_science_maxcoal30_robustness.py"),
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


if __name__ == "__main__":
    config = robustness_config()
    print(run_weak_full_core_robustness(config))
    print(run_lexicographic_full_core_robustness(config))
