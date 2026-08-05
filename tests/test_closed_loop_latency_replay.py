import unittest
from types import SimpleNamespace
from unittest.mock import patch

import dilu.runtime_support as runtime_support
from dilu.evaluation.factorial_replay import FACTORIAL_REPLAY_VERSION
from dilu.runtime_support import (
    _apply_closed_loop_latency_replay,
    _request_latency_contract,
    _resolve_latency_replay_delay,
)
from dilu.runtime_frame_trace import build_episode_event, classify_release_lifecycle

    def test_request_scoped_timing_fields_and_alignment_contract(self):
        cfg = {
            "policy_frequency": 10,
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.2,
                "delay_steps": 2,
                "target_systems": ["slow"],
            },
        }
        episode = {"action": 1, "latency_replay_queue": []}
        source_meta = {
            "system_used": "slow",
            "slow_request_attempted": True,
            "slow_request_valid_return": True,
            "closed_loop_latency_request_id": "scripted-request",
            "closed_loop_latency_response_outcome": "valid",
            "closed_loop_scripted_latency_steps": 1,
            "query_state_fast_proposal_action": 1,
        }

        _apply_closed_loop_latency_replay(
            frame=0,
            action=2,
            decision_meta=source_meta,
            episode_state=episode,
            cfg=cfg,
        )
        self.assertEqual(source_meta["closed_loop_latency_delay_steps"], 1)
        self.assertEqual(source_meta["closed_loop_latency_scheduled_steps"], 1)
        self.assertEqual(source_meta["closed_loop_latency_scheduled_release_frame"], 1)
        self.assertEqual(source_meta["closed_loop_latency_scheduled_seconds"], 0.1)
        self.assertEqual(source_meta["closed_loop_latency_realized_steps"], -1)
        self.assertTrue(source_meta["closed_loop_latency_realized_seconds"] != source_meta["closed_loop_latency_realized_seconds"])
        self.assertFalse(source_meta["closed_loop_latency_realized_available"])
        self.assertEqual(source_meta["closed_loop_latency_realized_source"], "not_released")
        self.assertTrue(source_meta["closed_loop_latency_scripted_sample"])
        self.assertFalse(source_meta["closed_loop_latency_contract_match"])
        self.assertIn("closed_loop_release_action_alignment_pass", source_meta)
        self.assertNotIn("closed_loop_release_action_alignment_passed", source_meta)

        terminal_meta = {"system_used": "fast"}
        _apply_closed_loop_latency_replay(
            frame=1,
            action=1,
            decision_meta=terminal_meta,
            episode_state=episode,
            cfg=cfg,
        )
        self.assertEqual(terminal_meta["closed_loop_latency_delay_steps"], 1)
        self.assertEqual(terminal_meta["closed_loop_latency_scheduled_release_frame"], 1)
        self.assertEqual(terminal_meta["closed_loop_latency_realized_steps"], 1)
        self.assertEqual(terminal_meta["closed_loop_latency_realized_seconds"], 0.1)
        self.assertTrue(terminal_meta["closed_loop_latency_realized_available"])
        self.assertEqual(terminal_meta["closed_loop_latency_realized_source"], "simulator_frame_delta")


class ClosedLoopLatencyReplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "policy_frequency": 10,
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.2,
                "delay_steps": 2,
                "target_systems": ["slow"],
            },
        }
        self.episode = {"action": 4, "latency_replay_queue": []}

    def apply(self, frame, action, system):
        meta = {"system_used": system}
        result = _apply_closed_loop_latency_replay(
            frame=frame,
            action=action,
            decision_meta=meta,
            episode_state=self.episode,
            cfg=self.cfg,
        )
        self.episode["action"] = result
        return result, meta

    def test_pending_interval_continues_current_fast_policy(self):
        action0, meta0 = self.apply(0, 3, "slow")
        action1, meta1 = self.apply(1, 1, "fast")
        self.assertEqual(action0, 4)
        self.assertEqual(action1, 1)
        self.assertTrue(meta0["closed_loop_latency_action_delayed"])
        self.assertTrue(meta1["closed_loop_latency_action_delayed"])

    def test_query_frame_executes_logged_fast_counterfactual(self):
        meta = {"system_used": "slow", "query_state_fast_proposal_action": 2}
        executed = _apply_closed_loop_latency_replay(
            frame=0,
            action=3,
            decision_meta=meta,
            episode_state=self.episode,
            cfg=self.cfg,
        )
        self.assertEqual(executed, 2)
        self.assertEqual(meta["closed_loop_latency_provisional_controller"], "matched_fast_policy")

    def test_release_records_execution_state_counterfactual(self):
        self.apply(0, 3, "slow")
        self.apply(1, 1, "fast")
        released, meta = self.apply(2, 1, "fast")
        self.assertEqual(released, 3)
        self.assertTrue(meta["closed_loop_latency_release_event"])
        self.assertEqual(meta["closed_loop_execution_state_fast_action"], 1)
        self.assertEqual(meta["closed_loop_released_slow_action"], 3)
        self.assertTrue(meta["closed_loop_release_route_divergence"])
        self.assertFalse(meta["closed_loop_route_preserved_divergent_release"])
        self.assertFalse(meta["closed_loop_release_actuation_distinct"])
        self.assertFalse(meta["closed_loop_final_returns_to_fast"])
        self.assertTrue(meta["release_selection_distinct"])
        self.assertEqual(
            meta["closed_loop_latency_terminal_outcome"],
            "distinct_actuation",
        )

    def test_request_id_survives_query_queue_and_release(self):
        query_meta = {
            "system_used": "slow",
            "slow_request_attempted": True,
            "slow_request_valid_return": True,
            "query_state_fast_proposal_action": 2,
        }
        query_action = _apply_closed_loop_latency_replay(
            frame=0,
            action=4,
            decision_meta=query_meta,
            episode_state=self.episode,
            cfg=self.cfg,
        )
        self.episode["action"] = query_action
        request_id = query_meta["closed_loop_latency_request_id"]

        self.apply(1, 1, "fast")
        release_meta = {"system_used": "fast"}
        _apply_closed_loop_latency_replay(
            frame=2,
            action=1,
            decision_meta=release_meta,
            episode_state=self.episode,
            cfg=self.cfg,
        )

        self.assertTrue(request_id)
        self.assertEqual(
            self.episode["latency_replay_queue"],
            [],
        )
        self.assertEqual(release_meta["closed_loop_latency_request_id"], request_id)
        self.assertTrue(release_meta["closed_loop_latency_release_event"])

    def test_aligned_final_actuation_is_compared_directly_with_release_fast_action(self):
        distinct = classify_release_lifecycle(
            release_event=True,
            release_fast_action=1,
            released_slow_action=4,
            executed_action=4,
            release_action_unavailable=False,
            release_alignment_evaluated=True,
            release_alignment_passed=True,
        )
        returned_to_fast = classify_release_lifecycle(
            release_event=True,
            release_fast_action=1,
            released_slow_action=4,
            executed_action=1,
            release_action_unavailable=False,
            release_alignment_evaluated=True,
            release_alignment_passed=True,
        )

        self.assertTrue(distinct["closed_loop_release_actuation_distinct"])
        self.assertTrue(distinct["closed_loop_kept_distinct_release"])
        self.assertFalse(returned_to_fast["closed_loop_release_actuation_distinct"])
        self.assertFalse(returned_to_fast["closed_loop_kept_distinct_release"])
        self.assertTrue(returned_to_fast["closed_loop_final_returns_to_fast"])

    def test_final_actuation_requires_evaluated_passing_alignment(self):
        common = {
            "release_event": True,
            "release_fast_action": 1,
            "released_slow_action": 4,
            "executed_action": 4,
            "release_action_unavailable": False,
        }

        not_evaluated = classify_release_lifecycle(
            **common,
            release_alignment_evaluated=False,
            release_alignment_passed=True,
        )
        failed = classify_release_lifecycle(
            **common,
            release_alignment_evaluated=True,
            release_alignment_passed=False,
        )

        self.assertFalse(not_evaluated["closed_loop_release_actuation_distinct"])
        self.assertFalse(not_evaluated["closed_loop_kept_distinct_release"])
        self.assertFalse(failed["closed_loop_release_actuation_distinct"])
        self.assertFalse(failed["closed_loop_kept_distinct_release"])

    def test_event_trace_ignores_hidden_assist_and_stale_distinct_flag(self):
        event = build_episode_event(
            frame=54,
            state={},
            decision_meta={
                "closed_loop_latency_release_event": True,
                "closed_loop_execution_state_fast_action": 1,
                "closed_loop_released_slow_action": 4,
                "closed_loop_latency_executed_action": 1,
                "closed_loop_release_action_alignment_evaluated": True,
                "closed_loop_release_action_alignment_pass": True,
                "closed_loop_release_actuation_distinct": True,
                "hidden_slower_brake_assist": True,
            },
            terminal_outcome={},
        )

        self.assertFalse(event["closed_loop_release_actuation_distinct"])
        self.assertFalse(event["closed_loop_kept_distinct_release"])
        self.assertTrue(event["closed_loop_final_returns_to_fast"])

    def test_modern_event_recomputes_selection_and_refuses_cross_stage_effect_label(self):
        event = build_episode_event(
            frame=54,
            state={},
            decision_meta={
                "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
                "closed_loop_latency_release_event": True,
                "closed_loop_released_slow_action": 4,
                "closed_loop_release_action_alignment_evaluated": True,
                "closed_loop_release_action_alignment_pass": True,
                "release_fast_comparator_action": 1,
                "release_selected_action": 4,
                "release_action_comparison_stage": (
                    "post_release_guard_and_frame_safety_pre_actuator_bridge"
                ),
                "release_selection_distinct": False,
                "final_actuator_action": 1,
                "final_actuator_action_stage": "post_environment_action_adapter",
            },
            terminal_outcome={},
        )

        self.assertTrue(event["release_selection_distinct"])
        self.assertTrue(event["release_selection_comparison_available"])
        self.assertFalse(event["closed_loop_release_actuation_distinct"])
        self.assertFalse(event["closed_loop_final_returns_to_fast"])
        self.assertTrue(event["closed_loop_post_latency_shield_rewrite"])
        self.assertFalse(
            event["closed_loop_release_actuation_comparison_available"]
        )
        self.assertFalse(
            event["closed_loop_release_actuation_supports_rollout_effect_claim"]
        )
        self.assertEqual(
            event["closed_loop_release_actuation_semantics"],
            "unavailable_without_same_stage_final_fast_counterfactual",
        )

    def test_modern_release_event_fails_closed_on_missing_or_invalid_stages(self):
        valid = {
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "closed_loop_latency_release_event": True,
            "release_fast_comparator_action": 1,
            "release_selected_action": 4,
            "release_action_comparison_stage": (
                "post_release_guard_and_frame_safety_pre_actuator_bridge"
            ),
            "final_actuator_action": 4,
            "final_actuator_action_stage": (
                "post_shared_actuator_bridge_pre_environment_step"
            ),
        }
        cases = (
            (
                {key: value for key, value in valid.items() if key != "final_actuator_action_stage"},
                "missing its explicit action-stage contract",
            ),
            (
                {**valid, "release_action_comparison_stage": "query_state"},
                "invalid release action comparison stage",
            ),
            (
                {**valid, "final_actuator_action_stage": "unknown_stage"},
                "invalid final actuator action stage",
            ),
            (
                {
                    **valid,
                    "final_actuator_action_stage": (
                        "latency_replay_output_pre_shared_actuator_bridge"
                    ),
                },
                "invalid final actuator action stage",
            ),
        )

        for metadata, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    build_episode_event(54, {}, metadata, {})

    def test_legacy_release_ignores_default_sentinel_stage_fields(self):
        event = build_episode_event(
            frame=54,
            state={},
            decision_meta={
                "closed_loop_latency_release_event": True,
                "closed_loop_execution_state_fast_action": 1,
                "closed_loop_released_slow_action": 4,
                "closed_loop_release_action_alignment_evaluated": True,
                "closed_loop_release_action_alignment_pass": True,
                "release_fast_comparator_action": -1,
                "release_selected_action": -1,
                "release_action_comparison_stage": "",
                "final_actuator_action": 1,
                "final_actuator_action_stage": "decision_trace_pre_actuator_bridge",
            },
            terminal_outcome={},
        )

        self.assertEqual(event["closed_loop_execution_state_fast_action"], 1)
        self.assertEqual(event["release_selected_action"], 1)
        self.assertFalse(event["release_selection_comparison_available"])
        self.assertFalse(event["release_selection_distinct"])
        self.assertTrue(event["closed_loop_final_returns_to_fast"])

    def test_unavailable_stale_release_returns_to_current_fast_action(self):
        class DummyEnv:
            unwrapped = None

            def __init__(self):
                self.unwrapped = self

            @staticmethod
            def get_available_actions():
                return [1, 2, 3]

        episode = {"action": 1, "latency_replay_queue": []}
        source_meta = {"system_used": "slow"}
        _apply_closed_loop_latency_replay(
            frame=0,
            action=4,
            decision_meta=source_meta,
            episode_state=episode,
            cfg=self.cfg,
        )
        _apply_closed_loop_latency_replay(
            frame=1,
            action=3,
            decision_meta={"system_used": "fast"},
            episode_state=episode,
            cfg=self.cfg,
        )
        release_meta = {"system_used": "fast"}
        executed = _apply_closed_loop_latency_replay(
            frame=2,
            action=3,
            decision_meta=release_meta,
            episode_state=episode,
            cfg=self.cfg,
            env=DummyEnv(),
            state={},
            safety_system=None,
        )
        self.assertEqual(executed, 3)
        self.assertTrue(release_meta["closed_loop_release_action_unavailable"])
        # Unavailable releases are reported separately from safety rewrites;
        # the two lifecycle channels must not be double-counted.
        self.assertFalse(release_meta["closed_loop_post_latency_shield_rewrite"])
        self.assertTrue(release_meta["closed_loop_final_returns_to_fast"])

    def test_unavailable_old_release_does_not_fall_back_to_same_frame_new_slow(self):
        class DummyEnv:
            unwrapped = None

            def __init__(self):
                self.unwrapped = self
                self.config = {"policy_frequency": 10}

            @staticmethod
            def get_available_actions():
                return [1, 2, 3]

        episode = {"action": 1, "latency_replay_queue": []}
        first = _apply_closed_loop_latency_replay(
            frame=0,
            action=4,
            decision_meta={"system_used": "slow", "query_state_fast_proposal_action": 2},
            episode_state=episode,
            cfg=self.cfg,
        )
        episode["action"] = first
        second = _apply_closed_loop_latency_replay(
            frame=1,
            action=1,
            decision_meta={"system_used": "fast"},
            episode_state=episode,
            cfg=self.cfg,
        )
        episode["action"] = second

        release_meta = {
            "system_used": "slow",
            "query_state_fast_proposal_action": 3,
        }
        executed = _apply_closed_loop_latency_replay(
            frame=2,
            action=4,
            decision_meta=release_meta,
            episode_state=episode,
            cfg=self.cfg,
            env=DummyEnv(),
            state={},
            safety_system=None,
        )

        self.assertEqual(executed, 3)
        self.assertEqual(release_meta["closed_loop_execution_state_fast_action"], 3)
        self.assertTrue(release_meta["closed_loop_release_action_unavailable"])
        self.assertTrue(release_meta["closed_loop_final_returns_to_fast"])
        self.assertTrue(release_meta["closed_loop_latency_action_delayed"])
        self.assertEqual(len(episode["latency_replay_queue"]), 1)
        self.assertEqual(episode["latency_replay_queue"][0]["source_frame"], 2)
        self.assertEqual(episode["latency_replay_queue"][0]["action"], 4)

    def test_delay_is_derived_from_seconds_at_final_environment_frequency(self):
        cfg = {
            "policy_frequency": 10,
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.26,
                "delay_steps": 99,
            },
        }
        env = SimpleNamespace(
            unwrapped=SimpleNamespace(config={"policy_frequency": 4})
        )

        resolved = _resolve_latency_replay_delay(cfg, env)

        self.assertEqual(resolved["steps"], 2)
        self.assertEqual(resolved["policy_frequency_hz"], 4.0)
        self.assertEqual(
            resolved["policy_frequency_source"],
            "env.unwrapped.config.policy_frequency",
        )
        self.assertEqual(resolved["configured_steps"], 99)
        self.assertFalse(resolved["configured_steps_consistent"])

    def test_invalid_latency_cannot_gain_immediate_slow_action_authority(self):
        cfg = {
            "policy_frequency": 10,
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": float("nan"),
                "delay_steps": 0,
                "target_systems": ["slow"],
            },
        }
        episode = {"action": 1, "latency_replay_queue": []}
        meta = {
            "system_used": "slow",
            "slow_request_attempted": True,
            "slow_request_valid_return": True,
            "query_state_fast_proposal_action": 2,
        }

        executed = _apply_closed_loop_latency_replay(
            frame=0,
            action=4,
            decision_meta=meta,
            episode_state=episode,
            cfg=cfg,
        )

        self.assertEqual(executed, 2)
        self.assertTrue(meta["closed_loop_latency_contract_rejected"])
        self.assertEqual(
            meta["closed_loop_latency_contract_rejection_reason"],
            "nonfinite_latency",
        )
        self.assertFalse(meta["closed_loop_latency_eligible"])
        self.assertEqual(episode["latency_replay_queue"], [])

    def test_final_frequency_contract_matches_gate_and_runtime_queue(self):
        cfg = {
            "policy_frequency": 10,
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.26,
                "delay_steps": 99,
                "target_systems": ["slow"],
            },
        }
        env = SimpleNamespace(
            unwrapped=SimpleNamespace(config={"policy_frequency": 4})
        )
        contract = _resolve_latency_replay_delay(cfg, env)
        episode = {"action": 1, "latency_replay_queue": []}
        gate_fields = {
            "rgd_latency_contract_version": contract["version"],
            "rgd_latency_scheduled_steps": contract["scheduled_steps"],
            "rgd_latency_policy_frequency_hz": contract["policy_frequency_hz"],
        }
        query_meta = {
            "system_used": "slow",
            "query_state_fast_proposal_action": 1,
            **gate_fields,
        }

        executed = _apply_closed_loop_latency_replay(
            frame=0,
            action=3,
            decision_meta=query_meta,
            episode_state=episode,
            cfg=cfg,
            env=env,
        )
        episode["action"] = executed

        self.assertEqual(episode["latency_replay_queue"][0]["release_frame"], 2)
        self.assertTrue(query_meta["closed_loop_latency_contract_match"])
        self.assertEqual(query_meta["closed_loop_latency_scheduled_seconds"], 0.5)

        _apply_closed_loop_latency_replay(
            frame=1,
            action=1,
            decision_meta={"system_used": "fast", **gate_fields},
            episode_state=episode,
            cfg=cfg,
            env=env,
        )
        release_meta = {"system_used": "fast", **gate_fields}
        _apply_closed_loop_latency_replay(
            frame=2,
            action=1,
            decision_meta=release_meta,
            episode_state=episode,
            cfg=cfg,
            env=env,
        )

        self.assertTrue(release_meta["closed_loop_latency_contract_match"])
        self.assertEqual(release_meta["closed_loop_latency_realized_steps"], 2)
        self.assertEqual(release_meta["closed_loop_latency_realized_seconds"], 0.5)

    def test_delayed_hidden_slower_is_queued_without_touching_vehicle_target(self):
        vehicle = SimpleNamespace(
            speed=20.0,
            speed_index=0,
            target_speed=20.0,
            target_speeds=[20.0, 25.0, 30.0],
        )
        unwrapped = SimpleNamespace(
            get_available_actions=lambda: [1, 3],
            vehicle=vehicle,
            config={"env_type": "highway-v0", "policy_frequency": 10},
        )
        env = SimpleNamespace(
            unwrapped=unwrapped,
            config={},
            spec=SimpleNamespace(id="highway-v0"),
        )
        sce = SimpleNamespace(
            ego=object(),
            getSurroundingVehicles=lambda _count: [],
            describe=lambda _frame, _vehicles: "test scene",
        )
        cfg = {
            "env_type": "highway-v0",
            "scenario_type": "highway",
            "render_mode": "none",
            "policy_frequency": 10,
            "hidden_slower_bridge": {"enable": True},
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.2,
                "delay_steps": 2,
                "target_systems": ["slow"],
            },
        }
        state = {
            "speed": 20.0,
            "front_dist": 6.0,
            "front_speed": 14.0,
            "ttc": 1.0,
            "thw": 0.3,
        }
        route_meta = {
            "system_used": "slow",
            "slow_reasoning_success": True,
            "query_state_fast_proposal_action": 1,
        }

        with (
            patch.object(runtime_support, "get_driving_state", return_value=state),
            patch.object(
                runtime_support,
                "build_frame_driving_state",
                return_value=SimpleNamespace(),
            ),
            patch.object(runtime_support, "inject_safety_cost_bridge"),
            patch.object(
                runtime_support,
                "resolve_agent_action",
                return_value=(4, "slow response", route_meta),
            ),
        ):
            frame_bundle = runtime_support.run_frame_protocol(
                frame=0,
                env=env,
                sce=sce,
                agent=SimpleNamespace(),
                obs=[0.0],
                prev_action=1,
                cfg=cfg,
                safety_system=None,
                phys_rec=None,
                reas_rec=None,
                history_buffer=[],
                prev_frame_image=None,
            )

        routed_action = frame_bundle[0]
        decision_meta = frame_bundle[-1]
        self.assertEqual(routed_action, 4)
        self.assertEqual(vehicle.target_speed, 20.0)

        episode = {"action": 1, "latency_replay_queue": []}
        executed = _apply_closed_loop_latency_replay(
            frame=0,
            action=routed_action,
            decision_meta=decision_meta,
            episode_state=episode,
            cfg=cfg,
            env=env,
            state=state,
            safety_system=None,
        )

        self.assertEqual(executed, 1)
        self.assertEqual(episode["latency_replay_queue"][0]["action"], 4)
        self.assertEqual(vehicle.target_speed, 20.0)

    def test_frame_zero_source_is_not_replaced_by_release_frame(self):
        event = build_episode_event(
            frame=17,
            state={},
            decision_meta={"closed_loop_latency_source_frame": 0},
            terminal_outcome={},
        )
        self.assertEqual(event["closed_loop_latency_source_frame"], 0)

    def test_trace_separates_predicted_scheduled_and_realized_latency(self):
        contract = _resolve_latency_replay_delay(self.cfg)
        gate_fields = {
            "rgd_latency_contract_version": contract["version"],
            "rgd_latency_scheduled_steps": contract["scheduled_steps"],
            "rgd_latency_policy_frequency_hz": contract["policy_frequency_hz"],
        }
        query_meta = {
            "system_used": "slow",
            "query_state_fast_proposal_action": 2,
            **gate_fields,
        }
        query_action = _apply_closed_loop_latency_replay(
            frame=0,
            action=3,
            decision_meta=query_meta,
            episode_state=self.episode,
            cfg=self.cfg,
        )
        self.episode["action"] = query_action

        self.assertEqual(query_meta["closed_loop_latency_predicted_seconds"], 0.2)
        self.assertEqual(query_meta["closed_loop_latency_scheduled_seconds"], 0.2)
        self.assertEqual(query_meta["closed_loop_latency_scheduled_steps"], 2)
        self.assertFalse(query_meta["closed_loop_latency_realized_available"])
        self.assertTrue(query_meta["closed_loop_latency_contract_match_available"])
        self.assertTrue(query_meta["closed_loop_latency_contract_match"])
        self.assertEqual(query_meta["closed_loop_latency_scheduled_release_frame"], 2)

        for frame in (1, 2):
            release_meta = {"system_used": "fast", **gate_fields}
            released = _apply_closed_loop_latency_replay(
                frame=frame,
                action=1,
                decision_meta=release_meta,
                episode_state=self.episode,
                cfg=self.cfg,
            )
            self.episode["action"] = released

        self.assertTrue(release_meta["closed_loop_latency_release_event"])
        self.assertTrue(release_meta["closed_loop_latency_realized_available"])
        self.assertEqual(release_meta["closed_loop_latency_realized_steps"], 2)
        self.assertEqual(release_meta["closed_loop_latency_realized_seconds"], 0.2)
        self.assertEqual(
            release_meta["closed_loop_latency_realized_source"],
            "simulator_frame_delta",
        )

        event = build_episode_event(
            frame=2,
            state={},
            decision_meta=release_meta,
            terminal_outcome={},
        )
        self.assertEqual(event["closed_loop_latency_predicted_seconds"], 0.2)
        self.assertEqual(event["closed_loop_latency_scheduled_seconds"], 0.2)
        self.assertEqual(event["closed_loop_latency_realized_seconds"], 0.2)
        self.assertEqual(event["closed_loop_latency_realized_steps"], 2)

    def test_zero_delay_replay_still_exports_the_resolved_contract(self):
        cfg = {
            "policy_frequency": 4,
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.0,
                "delay_steps": 9,
            },
        }
        meta = {"system_used": "slow"}

        executed = _apply_closed_loop_latency_replay(
            frame=0,
            action=3,
            decision_meta=meta,
            episode_state={"action": 1, "latency_replay_queue": []},
            cfg=cfg,
        )

        self.assertEqual(executed, 3)
        self.assertEqual(meta["closed_loop_latency_scheduled_steps"], 0)
        self.assertEqual(meta["closed_loop_latency_scheduled_seconds"], 0.0)
        self.assertFalse(meta["closed_loop_latency_realized_available"])
        self.assertFalse(meta["closed_loop_latency_configured_steps_consistent"])

    def test_request_level_jitter_changes_scheduling_not_latency_prediction(self):
        contract = _resolve_latency_replay_delay(self.cfg)
        sampled = _request_latency_contract(
            contract,
            {"closed_loop_scripted_latency_steps": 5},
        )
        self.assertEqual(sampled["predicted_steps"], 2)
        self.assertEqual(sampled["scheduled_steps"], 5)
        self.assertEqual(sampled["scheduled_seconds"], 0.5)
        self.assertEqual(
            sampled["source"],
            "decision_meta.closed_loop_scripted_latency_steps",
        )

    def test_scripted_latency_is_bound_to_its_request_through_release(self):
        query_meta = {
            "system_used": "slow",
            "slow_request_attempted": True,
            "slow_request_valid_return": True,
            "query_state_fast_proposal_action": 1,
            "closed_loop_scripted_latency_steps": 4,
        }
        _apply_closed_loop_latency_replay(
            frame=0,
            action=4,
            decision_meta=query_meta,
            episode_state=self.episode,
            cfg=self.cfg,
        )
        self.assertEqual(
            self.episode["latency_replay_queue"][0]["release_frame"], 4
        )
        for frame in (1, 2, 3, 4):
            meta = {"system_used": "fast"}
            _apply_closed_loop_latency_replay(
                frame=frame,
                action=1,
                decision_meta=meta,
                episode_state=self.episode,
                cfg=self.cfg,
            )
        self.assertTrue(meta["closed_loop_latency_release_event"])
        self.assertEqual(meta["closed_loop_latency_realized_steps"], 4)
        self.assertEqual(meta["closed_loop_latency_scheduled_steps"], 4)
        self.assertTrue(meta["closed_loop_latency_scripted_sample"])

    def test_invalid_scripted_latency_fails_closed(self):
        base = _resolve_latency_replay_delay(self.cfg)
        for invalid in (-1, float("nan"), float("inf"), "2.5", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    _request_latency_contract(
                        base,
                        {"closed_loop_scripted_latency_steps": invalid},
                    )


if __name__ == "__main__":
    unittest.main()
