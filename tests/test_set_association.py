"""Arithmetic and boundary tests for the explicit auxiliary association task."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from river_calibration import (
    ClutterFrameExposure, GroupProtocol, HistogramSpec, LabeledAppearanceBatch,
    ReservedGroups,
)
from river_set_association import (
    fit_representative_association, score_acceptable_set, simple_positive_selector,
)


class SetAssociationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = GroupProtocol(("fit",), ("validation",), ReservedGroups(("reserved",), "a"*64, "synthetic"))
        self.spec = HistogramSpec(("positive",), (np.array([0.0, 0.5, 1.0]),), 0.5, "synthetic score")
        # Two acceptable proposals retained in f1, null-with-candidate f2, empty f3.
        self.batch = LabeledAppearanceBatch(
            np.array([[0.1], [0.9], [0.8], [0.2]]),
            np.array([True, True, False, False]),
            ("fit",)*4, ("f1", "f1", "f1", "f2"), ("a", "b", "c", "d"),
        )
        self.exposure = ClutterFrameExposure(("f1", "f2", "f3"), ("fit",)*3, (1, 1, 0))

    def fit(self, batch=None, exposure=None, spec=None):
        return fit_representative_association(batch or self.batch, exposure or self.exposure,
                                             spec=spec or self.spec, protocol=self.protocol,
                                             data_id="fixture-v1")

    def test_duplicate_expected_sufficient_statistics_and_complete_cardinality(self):
        model = self.fit()
        self.assertEqual(model.metadata["fit_candidates"], 4)
        self.assertEqual(model.metadata["fit_duplicate_target_match_frames"], 1)
        self.assertEqual(model.metadata["fit_representative_effective_count"], 1)
        self.assertEqual(model.metadata["fit_nuisance_effective_count"], 3)
        self.assertAlmostEqual(model.expected_nuisance_count, 1)
        self.assertAlmostEqual(model.metadata["fit_physical_non_target_rate"], 2/3)
        self.assertAlmostEqual(model.candidate_present_prior_q, 1.5/4)
        np.testing.assert_allclose(model.target_bin_mass, [0.5, 0.5])
        np.testing.assert_allclose(model.nuisance_bin_mass, [0.5, 0.5])

    def test_bayes_normalization_matches_hand_arithmetic(self):
        model = self.fit()
        p = model.predict([[0.1], [0.9]])
        np.testing.assert_allclose(p, np.array([0.625, 0.375, 0.375])/1.375)
        np.testing.assert_allclose(model.predict(np.empty((0, 1))), [1])
        self.assertAlmostEqual(p.sum(), 1)

    def test_candidate_permutation_changes_only_matching_probability_positions(self):
        asymmetric = HistogramSpec(("positive",), (np.array([0, 0.15, 0.5, 1]),), 0.5, "asymmetric")
        model = self.fit(spec=asymmetric)
        a = model.predict([[0.1], [0.2], [0.9]])
        b = model.predict([[0.9], [0.1], [0.2]])
        np.testing.assert_allclose(b, a[[0, 3, 1, 2]])

    def test_positive_mass_and_read_only_arrays(self):
        model = self.fit()
        self.assertTrue(np.all(model.predict([[0], [1]]) > 0))
        with self.assertRaises(ValueError):
            model.target_bin_mass[0] = 1
        with self.assertRaises(ValueError):
            model.target_bin_mass.setflags(write=True)

    def test_deterministic_fingerprint_and_parameter_sensitivity(self):
        a, b = self.fit(), self.fit()
        self.assertEqual(a.metadata["model_sha256"], b.metadata["model_sha256"])
        alternative = HistogramSpec(("positive",), self.spec.bin_edges, 1, "synthetic score")
        self.assertNotEqual(a.metadata["model_sha256"], self.fit(spec=alternative).metadata["model_sha256"])

    def test_forbid_validation_and_reserved_training_rows(self):
        for forbidden in ("validation", "reserved", "unknown"):
            batch = LabeledAppearanceBatch(self.batch.scores, self.batch.is_target,
                                          (forbidden,)*4, self.batch.frame_ids, self.batch.sample_ids)
            with self.assertRaises(ValueError):
                self.fit(batch=batch)

    def test_missing_and_incorrect_exposure_rejected(self):
        for exposure in (ClutterFrameExposure(("f1", "f3"), ("fit",)*2, (1, 0)),
                         ClutterFrameExposure(("f1", "f2", "f3"), ("fit",)*3, (0, 1, 0))):
            with self.assertRaises(ValueError):
                self.fit(exposure=exposure)

    def test_no_target_or_no_nuisance_is_not_fabricated(self):
        for labels in (np.array([False]*4), np.array([True]*4)):
            batch = LabeledAppearanceBatch(self.batch.scores, labels, self.batch.group_ids, self.batch.frame_ids, self.batch.sample_ids)
            exposure = ClutterFrameExposure(("f1", "f2", "f3"), ("fit",)*3, (3, 1, 0) if not labels.any() else (0, 0, 0))
            if labels.all():
                # Duplicates provide nuisance, so all physical-positive data is
                # permitted by this auxiliary model, with no fake physical negatives.
                model = self.fit(batch=batch, exposure=exposure)
                self.assertEqual(model.metadata["fit_physical_non_target_rows"], 0)
                self.assertGreater(model.expected_nuisance_count, 0)
            else:
                with self.assertRaises(ValueError):
                    self.fit(batch=batch, exposure=exposure)

    def test_out_of_range_nonfinite_wrong_dim_rejected(self):
        model = self.fit()
        for scores in ([[1.000001]], [[-0.1]], [[float("nan")]], [[0.1, 0.2]], []):
            with self.assertRaises(ValueError):
                model.predict(scores)

    def test_set_and_uniform_scores_have_distinct_exact_meanings(self):
        s = score_acceptable_set([0.1, 0.2, 0.7], [1, 2])
        self.assertAlmostEqual(s["acceptable_set_nll"], -math.log(0.9))
        self.assertAlmostEqual(s["uniform_representative_nll"], -(math.log(0.2)+math.log(0.7))/2)
        self.assertAlmostEqual(s["expected_representative_onehot_brier"], 0.1**2+0.2**2+0.7**2-0.9+1)
        self.assertAlmostEqual(s["uniform_target_brier"], 0.1**2+0.3**2+0.2**2)
        self.assertAlmostEqual(s["candidate_present_brier"], 0.01)
        self.assertAlmostEqual(s["candidate_present_nll"], -math.log(0.9))
        self.assertGreater(s["duplicate_uniformity_excess"], 0)
        self.assertEqual(s["correct"], 1)

    def test_singleton_recovers_standard_categorical_nll_and_brier(self):
        for labels in ([0], [1]):
            s = score_acceptable_set([0.25, 0.75], labels)
            self.assertAlmostEqual(s["acceptable_set_nll"], s["uniform_representative_nll"])
            self.assertAlmostEqual(s["uniform_target_brier"], s["expected_representative_onehot_brier"])

    def test_all_error_types_and_zero_candidate_null(self):
        self.assertEqual(score_acceptable_set([0.8, 0.2], [1])["error_kind"], "null_on_target_match")
        self.assertEqual(score_acceptable_set([0.2, 0.8], [0])["error_kind"], "candidate_on_null")
        self.assertEqual(score_acceptable_set([0.1, 0.2, 0.7], [1])["error_kind"], "wrong_candidate_on_target_match")
        s = score_acceptable_set([1.0], [0])
        self.assertEqual(s["correct"], 1)
        self.assertEqual(s["acceptable_set_nll"], 0)

    def test_baseline_and_duplicate_do_not_hide_infinite_probability_loss(self):
        p = simple_positive_selector([[0.2], [0.8]])
        s = score_acceptable_set(p, [1, 2])
        self.assertEqual(s["acceptable_set_nll"], 0)
        self.assertTrue(math.isinf(s["uniform_representative_nll"]))
        self.assertEqual(s["correct"], 1)
        self.assertTrue(math.isinf(score_acceptable_set(p, [1])["acceptable_set_nll"]))
        np.testing.assert_equal(simple_positive_selector(np.empty((0, 1))), [1])

    def test_uniform_target_scores_are_minimized_by_uniform_representative(self):
        uniform = score_acceptable_set([0, 0.5, 0.5], [1, 2])
        concentrated = score_acceptable_set([0, 0.9, 0.1], [1, 2])
        # Set-event NLL cannot distinguish these; proper representative scores can.
        self.assertEqual(uniform["acceptable_set_nll"], concentrated["acceptable_set_nll"])
        self.assertLess(uniform["uniform_representative_nll"], concentrated["uniform_representative_nll"])
        self.assertLess(uniform["expected_representative_onehot_brier"], concentrated["expected_representative_onehot_brier"])

    def test_invalid_partial_labels_and_probabilities_rejected(self):
        for p, a in (([0.1, 0.9], []), ([0.1, 0.9], [0, 1]), ([0.1, 0.9], [1, 1]),
                     ([0.1, 0.9], [2]), ([0.1, 0.9], [True]), ([0.1, 0.8], [1]),
                     ([float("nan"), 0.9], [1])):
            with self.assertRaises(ValueError):
                score_acceptable_set(p, a)


if __name__ == "__main__":
    unittest.main()
