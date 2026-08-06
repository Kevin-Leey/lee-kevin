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
                "You are a driving decision module. Assess the current observation "
                "and select one available action. Return JSON only with final_action, "
                "confidence, and reason_lines.\n\n"
                "Observation:\n{observation}\n\n"
                "Available actions: {available_actions}"
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
