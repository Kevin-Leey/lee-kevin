import unittest
from unittest.mock import patch

from dilu.safety.rss_calculator import FASTER, IDLE, LANE_LEFT, SLOWER, RSSCalculator


class RSSTargetLaneEscapeTests(unittest.TestCase):
    def setUp(self):
        self.rss = RSSCalculator(
            {
                "policy_frequency": 10,
                "safety_thresholds": {"min_lane_gap": 7.0},
                "rss_params": {
                    "reaction_time": 0.25,
                    "max_accel": 1.0,
                    "min_brake_decel": 4.5,
                    "max_brake_decel": 4.5,
                    "safety_margin": 1.0,
                    "slack": 0.15,
                },
            }
        )

    @staticmethod
    def state_after_initial_braking():
        return {
            "scenario_type": "highway",
            "env_type": "highway-v0",
            "speed": 19.51721107681756,
            "front_dist": 7.622897693613538,
            "front_speed": 19.233070196353015,
            "ttc": float("inf"),
            "thw": 0.3905731030735214,
            "lane": 5,
            "total_lanes": 6,
            "has_left": True,
            "has_right": False,
            "left_front": 15.785340524616657,
            "left_rear": 100.0,
            "left_front_speed": 22.538938415502248,
            "left_rear_speed": float("nan"),
            "closest_vehicle_distance": 7.622897693613538,
            "closest_vehicle_longitudinal": 7.622897693613538,
            "closest_vehicle_lateral": 0.0,
            "closest_vehicle_closing_speed": 0.28414088046454467,
            "closest_vehicle_heading_delta": 0.0,
            "cross_traffic_dist": float("inf"),
        }

    def test_target_lane_with_faster_lead_and_clear_rear_is_released(self):
        action, _, overridden = self.rss.filter_action(
            LANE_LEFT,
            [LANE_LEFT, 1, 3, SLOWER],
            self.state_after_initial_braking(),
        )

        self.assertEqual(action, LANE_LEFT)
        self.assertFalse(overridden)

    def test_same_target_gap_is_blocked_before_speed_is_reduced(self):
        state = self.state_after_initial_braking()
        state.update(
            speed=26.0,
            front_dist=7.969421910075852,
            front_speed=23.433070196353025,
            ttc=3.1046512837060316,
            thw=0.3065162273106097,
            left_front=15.379806684373989,
            left_front_speed=22.841375073141307,
            closest_vehicle_distance=7.969421910075852,
            closest_vehicle_longitudinal=7.969421910075852,
            closest_vehicle_closing_speed=2.5669298036469748,
        )

        action, _, overridden = self.rss.filter_action(
            LANE_LEFT,
            [LANE_LEFT, 1, 3, SLOWER],
            state,
        )

        self.assertEqual(action, SLOWER)
        self.assertTrue(overridden)

    def test_clear_target_lane_is_not_blocked_by_same_lane_lead(self):
        state = self.state_after_initial_braking()
        state.update(
            speed=6.350134331140038,
            front_dist=5.5449769216786535,
            front_speed=4.989228669341223,
            ttc=4.074475606449772,
            thw=0.8732062398250346,
            lane=2,
            total_lanes=4,
            left_front=30.488473419155696,
            left_rear=100.0,
            left_front_speed=6.100995868201837,
            closest_vehicle_distance=5.545073489440813,
            closest_vehicle_longitudinal=5.54502389243631,
            closest_vehicle_lateral=-0.02345283799001549,
            closest_vehicle_closing_speed=1.3609357517528622,
        )

        action, _, overridden = self.rss.filter_action(
            LANE_LEFT,
            [LANE_LEFT, 1, 3, SLOWER],
            state,
        )

        self.assertEqual(action, LANE_LEFT)
        self.assertFalse(overridden)

    def test_escape_decision_is_stable_across_the_old_closing_speed_cutoff(self):
        state = self.state_after_initial_braking()
        state.update(
            speed=6.428451819607914,
            front_dist=5.502133772542379,
            front_speed=4.989228669341223,
            ttc=3.822988652956856,
            thw=0.8559034005294865,
            lane=2,
            total_lanes=4,
            left_front=30.44563027001942,
            left_rear=100.0,
            left_front_speed=6.100995868201837,
            closest_vehicle_distance=5.502231010948084,
            closest_vehicle_longitudinal=5.502182096250398,
            closest_vehicle_lateral=-0.023201126121953506,
            closest_vehicle_closing_speed=1.439223150266691,
        )

        action, _, overridden = self.rss.filter_action(
            LANE_LEFT,
            [LANE_LEFT, 1, 3, SLOWER],
            state,
        )

        self.assertEqual(action, LANE_LEFT)
        self.assertFalse(overridden)

        for closing_speed in (1.399, 1.401):
            perturbed = dict(state)
            perturbed["front_speed"] = perturbed["speed"] - closing_speed
            perturbed["ttc"] = perturbed["front_dist"] / closing_speed
            perturbed["closest_vehicle_closing_speed"] = closing_speed
            action, _, overridden = self.rss.filter_action(
                LANE_LEFT,
                [LANE_LEFT, 1, 3, SLOWER],
                perturbed,
            )
            self.assertEqual(action, LANE_LEFT)
            self.assertFalse(overridden)

    def test_tight_source_gap_is_decided_by_completion_margin(self):
        state = self.state_after_initial_braking()
        state.update(
            speed=6.428451819607914,
            front_dist=5.502133772542379,
            front_speed=4.227598310590762,
            ttc=2.500969896610172,
            thw=0.8559034005294865,
            lane=2,
            total_lanes=4,
            left_front=30.44563027001942,
            left_rear=100.0,
            left_front_speed=6.100995868201837,
            closest_vehicle_distance=5.502231010948084,
            closest_vehicle_longitudinal=5.502182096250398,
            closest_vehicle_lateral=-0.023201126121953506,
            closest_vehicle_closing_speed=2.200853509017152,
        )

        self.assertTrue(self.rss._highway_target_lane_escape_is_safe(state, LANE_LEFT, state["speed"]))

        state.update(
            front_dist=5.0,
            front_speed=0.0,
            ttc=5.0 / state["speed"],
            closest_vehicle_distance=5.0,
            closest_vehicle_longitudinal=5.0,
            closest_vehicle_closing_speed=state["speed"],
        )
        self.assertFalse(self.rss._highway_target_lane_escape_is_safe(state, LANE_LEFT, state["speed"]))

    def test_target_gap_must_remain_safe_at_lane_change_completion(self):
        state = self.state_after_initial_braking()
        state.update(
            speed=6.428451819607914,
            front_dist=5.502133772542379,
            front_speed=4.989228669341223,
            ttc=3.822988652956856,
            thw=0.8559034005294865,
            lane=2,
            total_lanes=4,
            left_front=12.1,
            left_rear=100.0,
            left_front_speed=6.0,
            closest_vehicle_distance=5.502231010948084,
            closest_vehicle_longitudinal=5.502182096250398,
            closest_vehicle_lateral=-0.023201126121953506,
            closest_vehicle_closing_speed=1.439223150266691,
        )

        self.assertFalse(self.rss._highway_target_lane_escape_is_safe(state, LANE_LEFT, state["speed"]))

    def test_acceleration_is_held_while_adjacent_vehicle_remains_close(self):
        state = {
            "scenario_type": "highway",
            "env_type": "highway-v0",
            "speed": 7.66497552114599,
            "front_dist": 31.53697081767001,
            "front_speed": 4.294967017638981,
            "ttc": 9.358127964618182,
            "thw": 4.114425509992356,
            "closest_vehicle_distance": 6.417483790247188,
            "closest_vehicle_longitudinal": 3.2866759503889083,
            "closest_vehicle_lateral": -5.511974183105413,
            "closest_vehicle_closing_speed": 1.0128224947137423,
            "closest_vehicle_heading_delta": 0.0,
            "cross_traffic_dist": float("inf"),
        }

        action, reason, overridden = self.rss.filter_action(
            FASTER,
            [IDLE, FASTER, SLOWER],
            state,
        )

        self.assertEqual(action, IDLE)
        self.assertTrue(overridden)
        self.assertEqual(reason, "RSS_DCBF_ADJACENT_ACCELERATION_HOLD")

    def test_hidden_brake_release_uses_same_adjacent_vehicle_guard(self):
        state = {
            "scenario_type": "highway",
            "env_type": "highway-v0",
            "speed": 5.197970625375187,
            "closest_vehicle_distance": 6.440645374898461,
            "closest_vehicle_longitudinal": 3.2937606223039775,
            "closest_vehicle_lateral": -5.534713525392318,
            "closest_vehicle_closing_speed": -0.5600121423921928,
        }

        self.assertTrue(RSSCalculator.highway_adjacent_acceleration_risk(state))

    def test_override_candidates_are_restricted_to_available_actions(self):
        available = [LANE_LEFT, IDLE, FASTER, SLOWER]
        with patch.object(
            self.rss,
            "_action_is_safe",
            side_effect=lambda _state, action: action == 2,
        ):
            action, reason, overridden = self.rss.filter_action(
                IDLE,
                available,
                {"scenario_type": "highway", "speed": 20.0},
            )

        self.assertIn(action, available)
        self.assertEqual(action, SLOWER)
        self.assertTrue(overridden)
        self.assertEqual(reason, "RSS_DCBF_EMERGENCY_BRAKE")


if __name__ == "__main__":
    unittest.main()
