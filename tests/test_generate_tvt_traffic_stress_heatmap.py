import copy
import unittest

import matplotlib.pyplot as plt

from tools.generate_tvt_traffic_stress_heatmap import (
    DISPLAY_LABELS,
    GROUP_ORDER,
    METRICS,
    default_input,
    draw_figure,
    metric_matrix,
    read_rows,
    validate_rows,
)


class TrafficStressFigureInputTests(unittest.TestCase):
    def test_authoritative_summary_has_explicit_rgd_absolute_matrices(self):
        rows = read_rows(default_input())
        cells, groups, indexed = validate_rows(rows)

        self.assertEqual(cells, [(4, 2.0), (4, 3.0), (5, 2.0), (5, 3.0), (6, 2.0), (6, 3.0)])
        self.assertEqual(groups, list(GROUP_ORDER))
        self.assertEqual([DISPLAY_LABELS[group] for group in groups], ["RGD (ours)", "TTC-risk", "Fast-only"])

        success = metric_matrix(cells, groups, indexed, METRICS[0])
        collision = metric_matrix(cells, groups, indexed, METRICS[1])
        distance = metric_matrix(cells, groups, indexed, METRICS[2])
        self.assertEqual(success.shape, (3, 6))
        self.assertEqual(collision.shape, (3, 6))
        self.assertEqual(distance.shape, (3, 6))
        self.assertAlmostEqual(success[0, 0], 100.0 * 29 / 30)
        self.assertAlmostEqual(collision[0, 1], 100.0 * 7 / 30)
        self.assertAlmostEqual(distance[0, 2], 629.205839846833)

        figure = draw_figure(cells, groups, indexed)
        try:
            heatmap_axes = [axis for axis in figure.axes if axis.get_title()]
            self.assertEqual(len(heatmap_axes), 3)
            for axis in heatmap_axes:
                labels = [tick.get_text() for tick in axis.get_yticklabels()]
                self.assertEqual(labels[0], "RGD (ours)")
                self.assertEqual(len(axis.patches), 6)
        finally:
            plt.close(figure)

    def test_missing_method_fails_closed(self):
        rows = [row for row in read_rows(default_input()) if row["group"] != "rgd_fixed_policy"]
        with self.assertRaisesRegex(RuntimeError, "Expected exactly methods"):
            validate_rows(rows)

    def test_count_rate_drift_fails_closed(self):
        rows = copy.deepcopy(read_rows(default_input()))
        rows[0]["success_rate"] = "0.5"
        with self.assertRaisesRegex(RuntimeError, "Success count/rate mismatch"):
            validate_rows(rows)

    def test_paired_difference_schema_is_not_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "missing columns"):
            read_rows(default_input().with_name("lane_density_paired_endpoints.csv"))


if __name__ == "__main__":
    unittest.main()
