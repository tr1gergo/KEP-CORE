"""Append selective frozen-floor lexicographic stabilization experiments.

Only markets whose recorded donor-free lexicographic allocation was exactly
certified blocked are targeted.  Existing rows are never overwritten: the new
procedure uses the distinct algorithm key ``lexicographic_floor_stabilization``.
The runner is resumable at the market/order level and reconstructs the frozen
study graphs, partitions, donor orders, and solver seeds exactly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from instance_analysis import load_instance
from KEP_functions import (
    build_compat_graph,
    build_graph_features,
    lexicographic_floor_core_search,
    make_partition,
    prepare_lexicographic_floor_baseline,
)
from management_science_simulations import (
    _flatten_result,
    make_donor_orders,
    stable_seed,
)


ALGORITHM = "lexicographic_floor_stabilization"
EVIDENCE_TYPE = "lexicographic_floors_aggregate_cut_heuristic_exact_final_check"
AUGMENTATION_ID = "lexicographic_frozen_floors_v1"
MANIFEST_NAME = "lexicographic_floor_augmentation_manifest.json"
SUMMARY_NAME = "lexicographic_floor_summary.csv"
CODE_FILES = (
    Path("KEP_functions.py"),
    Path("management_science_simulations.py"),
    Path("append_lexicographic_floor_results.py"),
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


def inspect_lexicographic_floor_targets(
    output_dirs: Sequence[str | Path],
) -> pd.DataFrame:
    """Report selective target and current augmentation counts without solving."""
    records: list[dict[str, object]] = []
    for folder in output_dirs:
        output_dir = Path(folder)
        rows = _read_jsonl(output_dir / "raw_results.jsonl")
        targets = _target_initial_rows(_source_rows(rows))
        augmentation = [row for row in rows if row.get("algorithm") == ALGORITHM]
        records.append(
            {
                "output_dir": output_dir.as_posix(),
                "target_markets": len(targets),
                "maximum_order_rows": len(targets) * 3,
                "existing_augmentation_rows": len(augmentation),
                "markets_with_any_augmentation_row": len(
                    set(str(row["market_id"]) for row in augmentation)
                ),
            }
        )
    return pd.DataFrame(records)


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


if __name__ == "__main__":
    for folder, expected in (
        (Path("results/management_science_revised"), 41),
        (Path("results/management_science_maxcoal09_with_heuristics"), 65),
    ):
        print(append_lexicographic_floor_results(folder, expected_target_markets=expected))
