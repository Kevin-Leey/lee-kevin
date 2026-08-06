import json
import re
from typing import Any, Dict, List, Optional
from dilu.driver_agent.base.state import ActionType

_STRUCTURED_DECISION_MAX_REASON_LINES = 3
_ACTION_TOKEN_MAP = {
    "lane_left": int(ActionType.LANE_LEFT),
    "left_lane": int(ActionType.LANE_LEFT),
    "idle": int(ActionType.IDLE),
    "keep_lane": int(ActionType.IDLE),
    "maintain_lane": int(ActionType.IDLE),
    "lane_right": int(ActionType.LANE_RIGHT),
    "right_lane": int(ActionType.LANE_RIGHT),
    "faster": int(ActionType.FASTER),
    "accelerate": int(ActionType.FASTER),
    "slower": int(ActionType.SLOWER),
    "decelerate": int(ActionType.SLOWER),
    "slow_down": int(ActionType.SLOWER),
    "brake": int(ActionType.SLOWER),
}

class LLMUtils:
    """Utility class for parsing actions from LLM responses."""

    @staticmethod
    def _coerce_action_value(value: Any) -> Optional[int]:
        """Resolve one action from either an integer-like value or a common action token."""
        if value is None or isinstance(value, bool):
            return None

        try:
            action = int(value)
        except (TypeError, ValueError):
            action = None
        if action is not None and 0 <= action <= 4:
            return int(action)

        normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        if not normalized:
            return None
        mapped_action = _ACTION_TOKEN_MAP.get(normalized)
        if mapped_action is not None:
            return int(mapped_action)
        return None

    @staticmethod
    def _validate_structured_payload(payload: Dict[str, Any], available: List[int]) -> Dict[str, Any]:
        """Validate one structured slow-path payload against the runtime schema."""
        action = LLMUtils._payload_action(payload)
        if action is None:
            return {
                "schema_valid": False,
                "schema_error": "missing_or_invalid_final_action",
                "action": None,
                "confidence": None,
                "reason_lines": [],
                "action_in_available": False,
            }

        confidence_value = payload.get("confidence")
        parsed_confidence = None
        if confidence_value is not None:
            try:
                parsed_confidence = float(confidence_value)
            except (TypeError, ValueError):
                return {
                    "schema_valid": False,
                    "schema_error": "invalid_confidence_type",
                    "action": int(action),
                    "confidence": None,
                    "reason_lines": [],
                    "action_in_available": bool(int(action) in list(available)),
                }
            if not 0.0 <= parsed_confidence <= 1.0:
                return {
                    "schema_valid": False,
                    "schema_error": "confidence_out_of_range",
                    "action": int(action),
                    "confidence": None,
                    "reason_lines": [],
                    "action_in_available": bool(int(action) in list(available)),
                }

        raw_reasons = payload.get("reason_lines", payload.get("reasons", []))
        if raw_reasons is None:
            raw_reasons = []
        elif isinstance(raw_reasons, str):
            raw_reasons = [raw_reasons]
        elif not isinstance(raw_reasons, list):
            return {
                "schema_valid": False,
                "schema_error": "invalid_reason_lines_type",
                "action": int(action),
                "confidence": parsed_confidence,
                "reason_lines": [],
                "action_in_available": bool(int(action) in list(available)),
            }

        reason_lines = [
            str(item).strip()
            for item in raw_reasons[:_STRUCTURED_DECISION_MAX_REASON_LINES]
            if str(item).strip()
        ]
        return {
            "schema_valid": True,
            "schema_error": "",
            "action": int(action),
            "confidence": parsed_confidence,
            "reason_lines": reason_lines,
            "action_in_available": bool(int(action) in list(available)),
        }

    @staticmethod
    def _extract_json_candidate(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON object from a model response."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        search_start = 0
        while True:
            brace_start = cleaned.find("{", search_start)
            if brace_start < 0:
                return None
            depth = 0
            for index in range(brace_start, len(cleaned)):
                char = cleaned[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[brace_start:index + 1]
                        try:
                            payload = json.loads(candidate)
                        except json.JSONDecodeError:
                            search_start = brace_start + 1
                            break
                        if isinstance(payload, dict):
                            return payload
                        search_start = brace_start + 1
                        break
            else:
                return None

    @staticmethod
    def _payload_action(payload: Dict[str, Any]) -> Optional[int]:
        """Resolve the final action field from a structured response payload."""
        for key in ("final_action", "action", "decision", "action_id"):
            value = payload.get(key)
            action = LLMUtils._coerce_action_value(value)
            if action is not None:
                return int(action)
        return None

    @staticmethod
    def parse_structured_decision(reasoning: str, available: List[int]) -> Dict[str, Any]:
        """Parse and validate the required structured slow-path response."""
        payload = LLMUtils._extract_json_candidate(reasoning)
        if payload is None:
            raise ValueError("slow structured decision JSON object not found")
        validated = LLMUtils._validate_structured_payload(payload, available)
        if not bool(validated.get("schema_valid", False)):
            raise ValueError(f"invalid slow structured decision schema: {validated.get('schema_error', 'unknown')}")
        return {
            "action": int(validated.get("action", ActionType.IDLE)),
            "confidence": validated.get("confidence"),
            "reason_lines": list(validated.get("reason_lines", [])),
            "structured": True,
            "parse_mode": "json_schema",
            "json_candidate_found": True,
            "schema_valid": True,
            "schema_error": "",
            "action_in_available": bool(validated.get("action_in_available", False)),
        }
