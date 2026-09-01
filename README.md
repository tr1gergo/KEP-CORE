# KEP-CORE paper simulations

This repository contains the simulation code, frozen XML instances, canonical
results, tables, and figures used in the paper.

## Run the complete study

Open `paper_simulations.ipynb` from the repository root and run its single code
cell. That cell calls one resumable pipeline and performs, in order:

1. cap-9 TU- and strong-core robustness checks for 5, 10, 20, and 30
   organizations;
2. full weak-core and lexicographic checks for 5 and 10 organizations, where
   cap 9 exhausts every potentially blocking coalition;
3. full weak-core and lexicographic checks for 20 and 30 organizations with
   cap 30;
4. consolidation into one canonical JSONL file;
5. a 1,200-second retry pass for every inconclusive lexicographic market; and
6. strict validation and regeneration of the paper tables and figures.

The pipeline checkpoints each solver call and skips completed work when
resumed. Its only switch is `RUN_ALL_SIMULATIONS` in the notebook. Leave it
`True` for a complete run; set it to `False` to validate the committed results
and regenerate tables and figures without solving the models again.

Do not run another Gurobi study at the same time. The optimization is a
multi-day computation. Runtime measurements, and potentially the identity of
tied optimal allocations, can vary with the machine, Gurobi version, and
thread scheduling.

## Reproducibility

The experiment fixes master seed `20260819`. The same deterministic seed
construction is used for:

- selection of the 30 base instances from the 100 committed candidates;
- induced compatibility-graph samples;
- organizational partitions;
- three donor orders per conditional search;
- algorithm random-number generators; and
- Gurobi solver seeds.

Thus a fresh run uses exactly the same markets and random draws as the saved
study. The committed canonical file is
`results/management_science_full_core/results.jsonl`. It contains 6,771 solver
records for 1,440 markets and has SHA-256
`f52c68498b99673921861e9ef16e1e5a925535fc0daa70bfd429c046203d6325`.
Byte-for-byte equality is not expected after a fresh run because the JSON also
records timestamps and measured runtimes; the seeded experimental inputs and
scientific result fields are invariant.

## Repository layout

- `paper_simulations.ipynb` — the single run/validate entry point.
- `paper_simulations.py` — all study design, checkpointing, consolidation,
  retry, validation, table, and figure code.
- `KEP_functions.py` — optimization models, cycle/chain enumeration, core
  separators, and donor-addition algorithms.
- `instance_analysis.py` — XML instance parser.
- `instances_large/` — all 100 generated 1,000-pair candidate instances and
  their generator configuration. All are needed because the frozen sample of
  30 is selected from this complete candidate set.
- `results/management_science_full_core/results.jsonl` — canonical paper data.
- `results/management_science_full_core/retry_checkpoint.jsonl` — long-limit
  calls incorporated into the canonical data.
- `results/management_science_full_core/tables/` — paper-ready summary tables.
- `figures/full_core_simulations/` — final paper figures.
- `tests/test_paper_simulations.py` — regression tests.

Temporary reproduction checkpoints are written below
`results/management_science_full_core/reproduction_work/` and are ignored by
Git.

## Environment and Jupyter kernel

The recorded environment used Python 3.13, PuLP 3.3.2, and Gurobi 13.0.2.

```powershell
py -3.13 -m venv .venv-ms
.\.venv-ms\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ms\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-ms\Scripts\python.exe -m ipykernel install --user --name kep-core --display-name "Python (KEP-CORE)"
```

Select `Python (KEP-CORE)` as the notebook kernel. A valid Gurobi license is
required. A Web License Service license also requires access to Gurobi's token
server.

## Validation

```powershell
.\.venv-ms\Scripts\python.exe -m unittest discover -s tests -v
```

Strict validation checks the frozen instance hashes, complete Cartesian
design, donor accounting, full-core evidence, lexicographic floors, retry
provenance, and canonical result counts.
