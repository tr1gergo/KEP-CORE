"""Canonical full-core study data, validation, and paper figures.

The primary experiment uses the full weak-core audit for every market:
coalition cap 9 for 5 and 10 organizations and cap 30 for 20 and 30
organizations.  The grand coalition cannot weakly block an allocation that
retains donor-free maximum coverage, so cap 9 is exhaustive for n=10.  The
same argument makes cap 4 exhaustive for the two n=5 fallback calls.

This module deliberately keeps the final data in one standards-compliant
JSONL file.  Its first record freezes protocol and source provenance; every
remaining record is one selected algorithm call.  Non-finite legacy solver
diagnostics are converted to null.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCHEMA_VERSION = "management_science_full_core_v1"
MASTER_SEED = 20260819
SOURCE_CAP4 = Path("results/management_science_revised/raw_results.jsonl")
SOURCE_CAP9 = Path(
    "results/management_science_maxcoal09_with_heuristics/raw_results.jsonl"
)
SOURCE_CAP30 = Path(
    "results/management_science_maxcoal30_robustness/raw_results.jsonl"
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

INITIAL_N5_FALLBACK = "genxml-60.xml|p500|n5|r0|d3"
FLOOR_N5_FALLBACK = "genxml-24.xml|p500|n5|r0|d3"

SELECTIVE_RETRY_TARGETS = {
    "genxml-11.xml|p500|n10|r1|d3": "initial_audit",
    "genxml-71.xml|p500|n20|r0|d3": "initial_audit",
    "genxml-39.xml|p500|n10|r0|d3": "floor_stabilization",
    "genxml-4.xml|p500|n10|r1|d3": "floor_stabilization",
    "genxml-46.xml|p500|n10|r1|d3": "floor_stabilization",
    "genxml-65.xml|p500|n20|r1|d3": "floor_stabilization",
}

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
    source_cap4: str | Path = SOURCE_CAP4,
    source_cap9: str | Path = SOURCE_CAP9,
    source_cap30: str | Path = SOURCE_CAP30,
) -> dict[str, Any]:
    """Create the one-file final dataset from the completed checkpoints."""
    output_path = Path(output_path)
    source_cap4 = Path(source_cap4)
    source_cap9 = Path(source_cap9)
    source_cap30 = Path(source_cap30)
    validate_frozen_instances()
    raw4 = _result_frame(source_cap4)
    raw9 = _result_frame(source_cap9)
    raw30 = _result_frame(source_cap30)
    initial4 = _initial_lexicographic_rows(raw4, raw4["market_id"].nunique())
    initial9 = _initial_lexicographic_rows(raw9, 1_440)
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

    # One initial lexicographic audit per market.  The cap-4 fallback resolves
    # a time-limited n=5 call and is mathematically full-core for n=5.
    primary_initial = pd.concat(
        [
            initial9[initial9["num_players"].isin([5, 10])],
            initial30,
        ],
        ignore_index=True,
    )
    primary_initial = primary_initial[
        ~primary_initial["market_id"].eq(INITIAL_N5_FALLBACK)
    ]
    primary_initial = pd.concat(
        [
            primary_initial,
            initial4[initial4["market_id"].eq(INITIAL_N5_FALLBACK)],
        ],
        ignore_index=True,
    )
    if len(primary_initial) != 1_440 or primary_initial["market_id"].nunique() != 1_440:
        raise ValueError("Primary initial lexicographic sample is incomplete")
    for _, row in primary_initial.iterrows():
        n = int(row["num_players"])
        source_path = source_cap4 if row["market_id"] == INITIAL_N5_FALLBACK else (
            source_cap9 if n <= 10 else source_cap30
        )
        cap = 4 if source_path == source_cap4 else (9 if n <= 10 else 30)
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
    floor = floor[~floor["market_id"].eq(FLOOR_N5_FALLBACK)]
    fallback_rows = raw4[
        raw4["algorithm"].eq("lexicographic_floor_stabilization")
        & raw4["market_id"].eq(FLOOR_N5_FALLBACK)
    ]
    floor = pd.concat([floor, fallback_rows], ignore_index=True)
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
        source_path = source_cap4 if row["market_id"] == FLOOR_N5_FALLBACK else (
            source_cap9 if n <= 10 else source_cap30
        )
        cap = 4 if source_path == source_cap4 else (9 if n <= 10 else 30)
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
            "n5_fallback_cap": 4,
            "integer_lexicographic_tiers_normalized": True,
        },
        "frozen_instances": [
            {"filename": filename, "sha256": digest}
            for filename, digest in FROZEN_INSTANCES
        ],
        "source_files": {
            path.as_posix(): sha256_file(path)
            for path in (source_cap4, source_cap9, source_cap30)
        },
        "code_files": {
            path.as_posix(): sha256_file(path)
            for path in (
                Path("KEP_functions.py"),
                Path("management_science_simulations.py"),
                Path("management_science_full_core.py"),
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


def retry_inconclusive_lexicographic_markets(
    canonical_path: str | Path = CANONICAL_PATH,
    checkpoint_path: str | Path = RETRY_PATH,
    *,
    time_limit_seconds: int = 1_200,
    solver_threads: int = 8,
) -> dict[str, Any]:
    """Selectively rerun the six unresolved primary lexicographic markets.

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
    from management_science_simulations import (
        _flatten_result,
        make_donor_orders,
        stable_seed,
    )

    canonical_path = Path(canonical_path)
    checkpoint_path = Path(checkpoint_path)
    _, calls = load_canonical_results(canonical_path)
    initial = calls[
        calls["procedure"].eq("lexicographic_rule")
        & calls["stage"].eq("initial_audit")
    ].set_index("market_id")
    completed = _retry_checkpoint_rows(checkpoint_path)
    started = perf_counter()
    new_rows = 0

    for target_index, (market_id, target_stage) in enumerate(
        SELECTIVE_RETRY_TARGETS.items(), start=1
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
                f"[{target_index}/{len(SELECTIVE_RETRY_TARGETS)}] {market_id}: "
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
                    f"[{target_index}/{len(SELECTIVE_RETRY_TARGETS)}] {market_id}: "
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
                f"[{target_index}/{len(SELECTIVE_RETRY_TARGETS)}] {market_id} "
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
    )
    return {
        "targets": len(SELECTIVE_RETRY_TARGETS),
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
) -> dict[str, Any]:
    """Promote complete retry targets and rebuild the canonical JSONL."""
    canonical_path = Path(canonical_path)
    checkpoint_path = Path(checkpoint_path)
    metadata, calls = load_canonical_results(canonical_path)
    retry = _retry_checkpoint_rows(checkpoint_path)
    promoted: set[str] = set()
    additions: list[dict[str, Any]] = []

    for market_id, target_stage in SELECTIVE_RETRY_TARGETS.items():
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
            if SELECTIVE_RETRY_TARGETS[market_id] == "floor_stabilization"
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
        "current_full_core_driver_sha256": sha256_file(
            "management_science_full_core.py"
        ),
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


if __name__ == "__main__":
    print(json.dumps(build_all_from_completed_results(), indent=2))
