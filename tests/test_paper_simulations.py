import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from KEP_functions import (
    GraphFeatures,
    Partition,
    _add_altruist_vertex,
    _normalize_lexicographic_tier_value,
    core_tu_simple,
    enumerate_cycles,
    lexicographic_floor_core_search,
    lexicographic_core_search,
    prepare_lexicographic_floor_baseline,
    separate_blocking_coalition,
    solve_lexicographic_cycle_cover,
)
from paper_simulations import (
    inconclusive_lexicographic_targets,
    make_donor_orders,
    paper_study_configs,
)


class ExactSeparationTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [0, 1]
        self.adj_out = {0: [1], 1: [0]}
        self.partition = Partition(
            owner_of={0: 0, 1: 1},
            vertices_of_player={0: [0], 1: [1]},
            players=[0, 1],
        )
        self.cycle_db = enumerate_cycles(
            self.vertices, self.adj_out, self.partition, 2
        )

    def test_empty_allocation_is_blocked_and_full_cycle_is_stable(self):
        for core_type in ("weak", "strong", "tu"):
            blocked = separate_blocking_coalition(
                set(),
                self.cycle_db,
                self.partition,
                2,
                2,
                core_type=core_type,
                solver="CBC",
            )
            stable = separate_blocking_coalition(
                {0},
                self.cycle_db,
                self.partition,
                2,
                2,
                core_type=core_type,
                solver="CBC",
            )
            self.assertTrue(blocked["blocking"])
            self.assertTrue(blocked["certified"])
            self.assertFalse(stable["blocking"])
            self.assertTrue(stable["certified"])

    def test_lexicographic_result_tracks_zero_introduced_and_used_donors(self):
        result = lexicographic_core_search(
            self.vertices,
            self.adj_out,
            self.partition,
            2,
            graph_features=GraphFeatures({}, {}),
            max_coal_size=2,
            solver="CBC",
            altruist_edges={},
            rng=np.random.default_rng(1),
        )
        self.assertTrue(result["final_in_core"])
        self.assertTrue(result["certified"])
        self.assertEqual(result["altruists_introduced"], 0)
        self.assertEqual(result["altruists_used"], 0)

    def test_integer_lexicographic_tiers_drop_solver_noise(self):
        self.assertEqual(
            _normalize_lexicographic_tier_value("cycle_count", 175.0000009606),
            175.0,
        )
        self.assertEqual(
            _normalize_lexicographic_tier_value("same_blood", 439.9999997334),
            440.0,
        )
        self.assertEqual(
            _normalize_lexicographic_tier_value("hard_match", 7.9412624825),
            7.9412624825,
        )

    def test_lexicographic_cover_cannot_reuse_an_altruist(self):
        altruist = 4
        partition = Partition(
            owner_of={0: 0, 1: 1},
            vertices_of_player={0: [0], 1: [1]},
            players=[0, 1],
        )
        cycle_db = enumerate_cycles(
            [0, 1, altruist],
            {
                0: [altruist],
                1: [altruist],
                altruist: [0, 1],
            },
            partition,
            2,
            altruist_vertex=[altruist],
        )

        selection, objectives, status = solve_lexicographic_cycle_cover(
            cycle_db,
            2,
            partition,
            solver="CBC",
        )

        self.assertEqual(status, 1)  # pulp.LpStatusOptimal
        self.assertEqual(objectives["transplants"], 1)
        self.assertEqual(len(selection), 1)
        self.assertEqual(
            sum(altruist in cycle_db.cycles[cid].vertices for cid in selection),
            1,
        )

    def test_frozen_lexicographic_floors_stabilize_without_reoptimizing_tiers(self):
        vertices = [0, 1, 2, 3, 4]
        adj_out = {
            0: [1],
            1: [2],
            2: [0, 3],
            3: [4],
            4: [2],
        }
        partition = Partition(
            owner_of={0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
            vertices_of_player={0: [0, 1], 1: [2, 3], 2: [4]},
            players=[0, 1, 2],
        )
        features = GraphFeatures(
            donor_bloodtype={
                0: "A",
                1: "A",
                2: "A",
                3: "B",
                4: "C",
                5: "B",
            },
            patient_bloodtype={0: "A", 1: "A", 2: "A", 3: "B", 4: "C"},
        )
        baseline = prepare_lexicographic_floor_baseline(
            vertices,
            adj_out,
            partition,
            3,
            features,
            max_coal_size=2,
            solver="CBC",
            solver_seed=17,
        )

        self.assertTrue(baseline["certified"])
        self.assertFalse(baseline["in_core"])
        self.assertEqual(baseline["objective_tiers"]["transplants"], 3)
        self.assertEqual(baseline["objective_tiers"]["same_blood"], 3)

        result = lexicographic_floor_core_search(
            vertices,
            adj_out,
            partition,
            3,
            features,
            max_coal_size=2,
            solver="CBC",
            altruist_edges={5: [3]},
            donor_order=[5],
            rng=np.random.default_rng(18),
            solver_seed=19,
            baseline=baseline,
        )

        self.assertTrue(result["certified"])
        self.assertTrue(result["final_in_core"])
        self.assertEqual(result["altruists_introduced"], 1)
        self.assertEqual(result["altruists_used"], 1)
        self.assertGreaterEqual(result["num_cuts"], 1)
        self.assertEqual(
            result["baseline_objective_tiers"], baseline["objective_tiers"]
        )
        for name, floor in baseline["objective_tiers"].items():
            self.assertGreaterEqual(result["objective_tiers"][name] + 1e-7, floor)


class SequentialTuTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [0, 1, 2]
        self.adj_out = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1],
        }
        self.partition = Partition(
            owner_of={0: 0, 1: 1, 2: 2},
            vertices_of_player={0: [0], 1: [1], 2: [2]},
            players=[0, 1, 2],
        )

    def test_tu_adds_one_donor_then_rechecks_core(self):
        result = core_tu_simple(
            self.vertices,
            self.adj_out,
            self.partition,
            2,
            max_coal_size=2,
            solver="CBC",
            altruist_edges={3: [2]},
            donor_order=[3],
            rng=np.random.default_rng(2),
        )
        self.assertTrue(result["in_core"])
        self.assertTrue(result["certified"])
        self.assertEqual(result["altruists_introduced"], 1)
        self.assertEqual(result["altruists_used"], 1)
        self.assertEqual(result["introduced_donor_ids"], [3])
        self.assertEqual(result["used_donor_ids"], [3])
        self.assertGreaterEqual(result["num_cuts"], 1)
        self.assertGreaterEqual(result["num_separation_solves"], 1)

    def test_no_donor_certifies_tu_infeasibility(self):
        result = core_tu_simple(
            self.vertices,
            self.adj_out,
            self.partition,
            2,
            max_coal_size=2,
            solver="CBC",
            altruist_edges={},
            donor_order=[],
            rng=np.random.default_rng(2),
        )
        self.assertFalse(result["in_core"])
        self.assertTrue(result["certified"])
        self.assertEqual(result["termination_reason"], "infeasible_after_all_donors")


class RandomizationTests(unittest.TestCase):
    def test_real_altruist_with_no_sampled_targets_stays_isolated(self):
        adj_out = {0: [], 1: []}
        _add_altruist_vertex(
            adj_out,
            [0, 1],
            2,
            [],
            rng=np.random.default_rng(9),
            real_vertices={0, 1},
        )

        self.assertEqual(adj_out[2], [])

    def test_synthetic_altruist_can_request_fallback_targets_explicitly(self):
        adj_out = {0: [], 1: []}
        _add_altruist_vertex(
            adj_out,
            [0, 1],
            2,
            [],
            rng=np.random.default_rng(9),
            real_vertices={0, 1},
            synthesize_targets_if_empty=True,
        )

        self.assertTrue(adj_out[2])

    def test_donor_orders_are_reproducible_and_do_not_depend_on_delta(self):
        kwargs = dict(
            altruist_edges={10: [1], 11: [2], 12: [3], 13: [4]},
            master_seed=19,
            instance_name="instance.xml",
            pool_size=100,
            num_players=10,
            partition_rep=1,
            repetitions=3,
        )
        first = make_donor_orders(**kwargs)
        second = make_donor_orders(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        for order in first:
            self.assertEqual(set(order), {10, 11, 12, 13})

class UnifiedPipelineTests(unittest.TestCase):
    def test_frozen_pipeline_scopes_primary_and_supplementary_runs(self):
        cap9, cap30 = paper_study_configs("results/test-work")
        self.assertEqual(cap9.master_seed, 20260819)
        self.assertEqual(cap9.max_coal_size, 9)
        self.assertEqual(cap9.num_players, (5, 10, 20, 30))
        self.assertEqual(cap9.primary_num_players, (5, 10))
        self.assertEqual(cap30.master_seed, 20260819)
        self.assertEqual(cap30.max_coal_size, 30)
        self.assertEqual(cap30.num_players, (20, 30))

        default_cap9, default_cap30 = paper_study_configs()
        temporary_root = Path(tempfile.gettempdir()).resolve()
        self.assertTrue(Path(default_cap9.output_dir).resolve().is_relative_to(temporary_root))
        self.assertTrue(Path(default_cap30.output_dir).resolve().is_relative_to(temporary_root))

    def test_retry_targets_are_derived_from_inconclusive_rows(self):
        calls = pd.DataFrame(
            [
                {
                    "analysis_role": "primary_full_core",
                    "procedure": "lexicographic_rule",
                    "stage": "initial_audit",
                    "market_id": "unresolved",
                    "certified": False,
                    "in_core": False,
                },
                {
                    "analysis_role": "primary_full_core",
                    "procedure": "lexicographic_rule",
                    "stage": "initial_audit",
                    "market_id": "blocked",
                    "certified": True,
                    "in_core": False,
                },
                {
                    "analysis_role": "primary_full_core",
                    "procedure": "lexicographic_rule",
                    "stage": "floor_stabilization",
                    "market_id": "blocked",
                    "certified": False,
                    "in_core": False,
                },
                {
                    "analysis_role": "primary_full_core",
                    "procedure": "lexicographic_rule",
                    "stage": "initial_audit",
                    "market_id": "stable",
                    "certified": True,
                    "in_core": True,
                },
            ]
        )
        self.assertEqual(
            inconclusive_lexicographic_targets(calls),
            {"blocked": "floor_stabilization", "unresolved": "initial_audit"},
        )


if __name__ == "__main__":
    unittest.main()
