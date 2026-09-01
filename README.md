# KEP-CORE: full-core kidney-exchange simulations

This repository contains the code, generated XML instances, validated results,
tables, and figures used for the paper's final simulation study. The study
focuses on weak-core existence and the weak-core stability of a lexicographic
operational allocation. TU- and strong-core results are supplementary
robustness checks.

## Reproduction entry point

Run `paper_simulations_full_core.ipynb` from the repository root. It is the
only study notebook needed for the paper. The notebook performs the complete
pipeline in this order:

1. run the seeded cap-four and cap-nine base studies;
2. run floor-preserving lexicographic stabilization only for initially blocked
   lexicographic allocations;
3. run the full-coalition weak-core and lexicographic studies for 20 and 30
   organizations;
4. consolidate the selected scientific records into one canonical JSONL file;
5. rerun the hard lexicographic markets with a longer MIP limit; and
6. strictly validate the design and regenerate all paper tables and figures.

The notebook defaults to `RUN_FULL_REPRODUCTION = False`. In this mode it
validates the committed canonical data and regenerates the tables and figures
without rerunning the multi-day optimization study. Set the flag to `True`
only for an intentional clean or resumed reproduction. Intermediate
checkpoints are written under
`results/management_science_full_core/reproduction_work/` and are ignored by
Git.

Do not run another Gurobi study at the same time. The pipeline is append-only
and resumable, but runtime measurements and the identity of tied optima can
vary across machines, especially with multithreaded Gurobi.

## Repository contents

- `KEP_functions.py`: cycle and chain enumeration, core separators, TU-core
  cutting planes, weak/strong search heuristics, and lexicographic allocation
  with floor-preserving stabilization.
- `management_science_simulations.py`: seeded, resumable cap-four/cap-nine
  study driver.
- `append_lexicographic_floor_results.py`: conditional lexicographic-floor
  stage for the cap-four/cap-nine study.
- `management_science_maxcoal30_robustness.py`: full-coalition weak-core and
  lexicographic driver for 20 and 30 organizations.
- `management_science_full_core.py`: frozen-instance validation, canonical
  consolidation, selective long-limit retries, strict data checks, statistical
  summaries, and paper figures.
- `instance_analysis.py`: XML instance parser.
- `instances_large/`: all 100 generated 1,000-pair XML base instances and the
  generator configuration. All 100 files are retained because the frozen
  seeded selection of 30 instances is made from this complete candidate list.
- `results/management_science_full_core/results.jsonl`: unified canonical
  result calls used in the paper.
- `results/management_science_full_core/retry_checkpoint.jsonl`: the saved
  long-limit retry calls promoted into the canonical file.
- `results/management_science_full_core/tables/`: regenerated analysis tables
  and their manifest.
- `results/management_science_maxcoal30_robustness/`: retained raw cap-30
  checkpoint and protocol files for provenance.
- `figures/full_core_simulations/`: the three final paper figures.
- `tests/test_management_science_simulations.py`: unit and regression tests.

The canonical JSONL is validated against the frozen 30 instance names and
SHA-256 hashes, the complete Cartesian experimental design, donor accounting,
full-core evidence, lexicographic objective floors, and selective retry
provenance.

## Environment and Jupyter kernel

The recorded environment used Python 3.13, PuLP 3.3.2, and Gurobi 13.0.2.
Create a local environment and notebook kernel with:

```powershell
py -3.13 -m venv .venv-ms
.\.venv-ms\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ms\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-ms\Scripts\python.exe -m ipykernel install --user --name kep-core --display-name "Python (KEP-CORE)"
```

Select `Python (KEP-CORE)` as the notebook kernel. A valid Gurobi license is
required for the full study. A Web License Service license also requires
network access to Gurobi's token server.

## Experimental design

The frozen design uses:

- 30 seeded base instances selected from the 100 committed XML candidates;
- induced pools of 100, 200, and 500 patient--donor pairs;
- 5, 10, 20, and 30 organizations;
- cycle-length caps of two and three;
- two organizational partitions per graph and organization count;
- up to three seeded donor orders when order can affect a conditional search;
  and
- 1,440 market cells in total.

The primary weak-core and lexicographic analyses test every proper coalition.
For 20 and 30 organizations, the separator also includes the grand coalition.
For five and ten organizations, the grand coalition cannot block because each
candidate allocation preserves at least the donor-free maximum number of real
transplants. The primary results therefore provide full weak-core evidence at
every organization count. The supplementary TU and strong procedures use a
coalition cap of nine.

## Saved result check

The committed canonical data contain 6,771 result-call records for 1,440
markets. Strict validation gives:

- certified weak-core witnesses in all 1,440 markets: 1,439 donor-free and one
  donor-assisted;
- 1,369 initially stable and 71 initially blocked donor-free lexicographic
  allocations; and
- after floor-preserving stabilization, 1,381 donor-free and 59
  donor-assisted lexicographic outcomes, with no unresolved markets.

The one donor-assisted weak-core market is not proof that its donor-free weak
core is empty: the weak procedure is a search heuristic whose returned
allocations receive exact final verification. Similarly, introduced donor
prefixes from heuristic searches are operational upper bounds, not proofs of
minimum donor requirements.

## Tests and validation

Run the tests with:

```powershell
.\.venv-ms\Scripts\python.exe -m unittest discover -s tests -v
```

Run the final notebook with `RUN_FULL_REPRODUCTION = False` to perform strict
canonical validation and regenerate the committed tables and figures.
