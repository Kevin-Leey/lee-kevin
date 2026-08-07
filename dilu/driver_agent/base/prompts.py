"""Compact prompt templates for the structured slow driving reasoner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: str

    def render(self, **values: Any) -> str:
        return self.template.format(**values)


class PromptManager:
    """Provide the small, fixed prompt surface used by ``SlowThinker``."""

    _TEMPLATES: Dict[str, PromptTemplate] = {
        "slow_decision": PromptTemplate(
            name="slow_decision",
            template=(
                "Select one available driving action. Output exactly one compact JSON "
                "object using this schema: "
                '{{"final_action":1,"confidence":0.9,"reason_lines":["short reason"]}}. '
                "The final_action must be one of {available_actions}. Use one reason "
                "of at most eight words.\nObservation: {observation}"
            ),
        )
    }

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        try:
            return cls._TEMPLATES[name]
        except KeyError as exc:
            raise KeyError(f"unknown prompt template: {name}") from exc

    @classmethod
    def render(cls, name: str, **values: Any) -> str:
        return cls.get(name).render(**values)
