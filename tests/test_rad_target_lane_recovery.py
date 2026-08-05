import unittest

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.reasoning.rad import RADSignalController


class RADTargetLaneRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rad = RADSignalController({"target_lane_projection_enable": True})
        self.blocked_lane_payload = {
            "coupled_risk": 0.95,
            "rollout_risk": 0.95,
            "rollout_utility": 0.05,
        }

    def recovery_cost(self, state: DrivingState, action: ActionType):
        return self.rad.estimate_action_recovery_cost(
            state,
            int(action),
            self.blocked_lane_payload,
            ttc_pressure=0.95,
        )

    def test_clear_target_lane_costs_less_than_idle_when_current_lane_is_blocked(self):
        state = DrivingState(
            ego_speed=6.35,
            front_distance=5.54,
            front_speed=4.99,
            ttc=4.07,
            thw=0.87,
            ego_lane=0,
            total_lanes=4,
            can_change_right=True,
            right_front_distance=30.49,
            right_rear_distance=100.0,
            right_front_speed=6.10,
            right_rear_speed=6.35,
            legal_actions=[int(ActionType.IDLE), int(ActionType.LANE_RIGHT), int(ActionType.SLOWER)],
            scenario_type="highway",
        )

        idle_cost, _ = self.recovery_cost(state, ActionType.IDLE)
        lane_cost, parts = self.recovery_cost(state, ActionType.LANE_RIGHT)

        self.assertLess(lane_cost, idle_cost)
        self.assertLess(lane_cost, 0.55)
        self.assertEqual(parts["target_lane_projection_applied"], 1.0)
        self.assertEqual(parts["target_lane_available"], 1.0)
        self.assertAlmostEqual(parts["coupled_risk"], parts["target_lane_risk"])
        self.assertLess(parts["target_lane_risk"], self.blocked_lane_payload["coupled_risk"])

    def test_urgent_target_rear_vehicle_keeps_lane_action_fail_closed(self):
        state = DrivingState(
            ego_speed=20.0,
            front_distance=7.0,
            front_speed=10.0,
            ttc=0.7,
            thw=0.35,
            ego_lane=1,
            total_lanes=4,
            can_change_left=True,
            left_front_distance=60.0,
            left_rear_distance=12.0,
            left_front_speed=22.0,
            left_rear_speed=35.0,
            legal_actions=[int(ActionType.LANE_LEFT), int(ActionType.IDLE), int(ActionType.SLOWER)],
            scenario_type="highway",
        )

        idle_cost, _ = self.recovery_cost(state, ActionType.IDLE)
        lane_cost, parts = self.recovery_cost(state, ActionType.LANE_LEFT)

        self.assertEqual(parts["target_lane_available"], 1.0)
        self.assertEqual(parts["target_rear_urgent"], 1.0)
        self.assertLess(parts["target_rear_ttc"], 3.0)
        self.assertEqual(lane_cost, 1.0)
        self.assertGreaterEqual(lane_cost, idle_cost)
        self.assertEqual(parts["dominant_term_name"], "target_rear_urgent_penalty")

    def test_unavailable_target_lane_is_fail_closed_even_when_geometrically_clear(self):
        state = DrivingState(
            ego_speed=18.0,
            front_distance=8.0,
            front_speed=12.0,
            ttc=1.33,
            ego_lane=0,
            total_lanes=3,
            can_change_left=False,
            left_front_distance=80.0,
            left_rear_distance=80.0,
            left_front_speed=18.0,
            left_rear_speed=18.0,
            legal_actions=[int(ActionType.IDLE), int(ActionType.SLOWER)],
            scenario_type="highway",
        )

        lane_cost, parts = self.recovery_cost(state, ActionType.LANE_LEFT)

        self.assertEqual(parts["target_lane_available"], 0.0)
        self.assertEqual(parts["lane_commit_penalty"], 1.0)
        self.assertEqual(lane_cost, 1.0)

    def test_projection_disabled_fallback_still_keeps_action_costs_identifiable(self):
        state = DrivingState(
            ego_speed=6.35,
            front_distance=5.54,
            front_speed=4.99,
            ttc=4.07,
            ego_lane=0,
            total_lanes=4,
            can_change_right=True,
            right_front_distance=30.49,
            right_rear_distance=100.0,
            right_front_speed=6.10,
            right_rear_speed=6.35,
            legal_actions=[int(ActionType.IDLE), int(ActionType.LANE_RIGHT), int(ActionType.SLOWER)],
            scenario_type="highway",
        )
        legacy = RADSignalController()

        idle_cost, _ = legacy.estimate_action_recovery_cost(
            state,
            int(ActionType.IDLE),
            self.blocked_lane_payload,
            ttc_pressure=0.95,
        )
        lane_cost, parts = legacy.estimate_action_recovery_cost(
            state,
            int(ActionType.LANE_RIGHT),
            self.blocked_lane_payload,
            ttc_pressure=0.95,
        )

        self.assertGreater(lane_cost, idle_cost)
        self.assertEqual(parts["raw_cost_formula"], "common_risk_plus_residual_action_penalty")
        self.assertEqual(parts["residual_action_penalty_source"], "rad_action_semantics")
        self.assertNotIn("target_lane_projection_applied", parts)
        self.assertFalse(legacy.target_lane_projection_enable)


if __name__ == "__main__":
    unittest.main()
