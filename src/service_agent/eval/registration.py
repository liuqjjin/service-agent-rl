"""Register the governed agent arms with tau2's registry.

Arms are separate registry names because build_agent's factory contract has
no channel for arbitrary config: the TextRunConfig `agent` field is the arm
selector. The factory also receives the task, which is used for exactly one
thing -- stamping the audit trail for offline analysis. It never reaches the
model; the leak tests police every model-visible surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tau2.registry import registry

from service_agent.agent.governed import GovernanceConfig, GovernedLLMAgent

# Batch runs set this before calling run_tasks; each agent instance dumps its
# audit trail here when the orchestrator stops it.
_audit_dir: Optional[Path] = None


def set_audit_dir(path: Optional[Path]) -> None:
    global _audit_dir
    _audit_dir = Path(path) if path is not None else None


def _make_factory(config: GovernanceConfig):
    def factory(tools, domain_policy, **kwargs):
        agent = GovernedLLMAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
            config=config,
        )
        task = kwargs.get("task")
        if task is not None:
            agent.audit.task_id = task.id
        agent.audit_dir = _audit_dir
        return agent

    return factory


H1_AGENT = "governed_llm_agent_h1"
H2_AGENT = "governed_llm_agent_h2"


def register_governed_agents() -> None:
    """Idempotent registration of both governance arms."""
    existing = set(registry.get_agents())
    if H1_AGENT not in existing:
        registry.register_agent_factory(_make_factory(GovernanceConfig.h1()), H1_AGENT)
    if H2_AGENT not in existing:
        registry.register_agent_factory(_make_factory(GovernanceConfig.h2()), H2_AGENT)
