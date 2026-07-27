"""Governance metrics, computable for every arm -- including H0.

The trick that makes the ablation comparable: the same governor that gates
H1/H2 live is replayed *offline* over any trajectory. For H0 that answers
"how many executed writes would the gate have rejected" (the policy-violation
measure) without a gate ever having run; for H1/H2 it should find nothing,
because the live gate already refused those candidates. Offline replay uses
identical rule code, so the arms are measured with one yardstick.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results

from service_agent.governance.core import Decision, ProposedAction, idempotency_key
from service_agent.governance.evidence import EvidenceExtractor
from service_agent.governance.telecom_rules import WRITE_TOOLS, TelecomGovernor


@dataclass
class TrajectoryAnalysis:
    write_candidates: int = 0
    executed_writes: int = 0
    unauthorized_executed_writes: int = 0
    duplicate_side_effects: int = 0
    errored_tool_calls: int = 0
    verdicts: Counter = field(default_factory=Counter)
    unauthorized_reasons: Counter = field(default_factory=Counter)


def analyze_trajectory(messages: Iterable) -> TrajectoryAnalysis:
    """Replay one official trajectory through the governor, in order.

    Each assistant tool call is adjudicated against the evidence available at
    that moment, then the call and its result update the evidence -- the same
    information flow the live gate sees.
    """
    session = "offline"
    governor = TelecomGovernor(session_id=session)
    extractor = EvidenceExtractor()
    analysis = TrajectoryAnalysis()

    pending: dict[str, tuple[str, dict, object]] = {}  # call_id -> (name, args, verdict)

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            if msg.is_tool_call():
                for tc in msg.tool_calls:
                    proposed = ProposedAction.from_tool_call(tc)
                    verdict = governor.evaluate(proposed, extractor.state)
                    if proposed.tool_name in WRITE_TOOLS:
                        analysis.write_candidates += 1
                        analysis.verdicts[verdict.decision.value] += 1
                    pending[tc.id] = (proposed.tool_name, proposed.arguments, verdict)
            extractor.observe_agent_message(msg)
        elif isinstance(msg, UserMessage):
            if not msg.is_tool_call():
                extractor.observe_user_message(msg)
        elif isinstance(msg, ToolMessage):
            entry = pending.pop(msg.id, None)
            if entry is None:
                continue
            tool_name, arguments, verdict = entry
            if msg.error:
                analysis.errored_tool_calls += 1
                extractor.observe_tool_result(msg)
                continue
            key = ""
            if tool_name in WRITE_TOOLS:
                analysis.executed_writes += 1
                if verdict.decision is not Decision.ALLOW:
                    analysis.unauthorized_executed_writes += 1
                    analysis.unauthorized_reasons[verdict.reason_code] += 1
                if verdict.decision is Decision.DUPLICATE:
                    analysis.duplicate_side_effects += 1
                key = idempotency_key(session, tool_name, arguments)
            extractor.observe_tool_result(msg, idempotency_key=key)

    return analysis


@dataclass
class ArmMetrics:
    simulations: int
    avg_reward: float
    pass_hat_ks: dict[int, float]
    termination_reasons: dict[str, int]
    avg_agent_cost: float | None
    avg_user_cost: float | None
    # governance (offline replay, same yardstick for every arm)
    write_candidates: int
    executed_writes: int
    unauthorized_executed_writes: int
    unauthorized_write_rate: float  # per simulation
    duplicate_side_effects: int
    unauthorized_reasons: dict[str, int]
    errored_tool_calls: int


def summarize_results(results: Results) -> ArmMetrics:
    from tau2.metrics.agent_metrics import compute_metrics

    official = compute_metrics(results)
    totals = TrajectoryAnalysis()
    sims_with_violation = 0
    terminations: Counter = Counter()

    for sim in results.simulations:
        terminations[str(sim.termination_reason)] += 1
        analysis = analyze_trajectory(sim.messages)
        totals.write_candidates += analysis.write_candidates
        totals.executed_writes += analysis.executed_writes
        totals.unauthorized_executed_writes += analysis.unauthorized_executed_writes
        totals.duplicate_side_effects += analysis.duplicate_side_effects
        totals.errored_tool_calls += analysis.errored_tool_calls
        totals.verdicts.update(analysis.verdicts)
        totals.unauthorized_reasons.update(analysis.unauthorized_reasons)
        if analysis.unauthorized_executed_writes:
            sims_with_violation += 1

    n = len(results.simulations) or 1
    return ArmMetrics(
        simulations=len(results.simulations),
        avg_reward=official.avg_reward,
        pass_hat_ks=dict(official.pass_hat_ks or {}),
        termination_reasons=dict(terminations),
        avg_agent_cost=official.avg_agent_cost,
        avg_user_cost=getattr(official, "avg_user_cost", None),
        write_candidates=totals.write_candidates,
        executed_writes=totals.executed_writes,
        unauthorized_executed_writes=totals.unauthorized_executed_writes,
        unauthorized_write_rate=sims_with_violation / n,
        duplicate_side_effects=totals.duplicate_side_effects,
        unauthorized_reasons=dict(totals.unauthorized_reasons),
        errored_tool_calls=totals.errored_tool_calls,
    )


def summarize_results_file(path: Path) -> ArmMetrics:
    return summarize_results(Results.load(path))


def audit_summary(audit_dir: Path) -> dict:
    """Aggregate the live gate's own records (H1/H2 only): what it saw and
    what it refused, including candidates that never became trajectory
    messages and are therefore invisible to the offline replay."""
    decisions: Counter = Counter()
    normalizations: Counter = Counter()
    reasons: Counter = Counter()
    regenerations = 0
    sessions = 0
    for file in sorted(audit_dir.glob("audit_*.jsonl")):
        sessions += 1
        for line in file.read_text().splitlines():
            rec = json.loads(line)
            if rec["reason_code"] == "mixed_text_stripped":
                # The sanitizer records the formatting normalization, then the
                # main loop records the actual gate verdict for the same
                # candidate. It is an event, not a second allow decision.
                normalizations[rec["reason_code"]] += 1
                continue
            decisions[rec["decision"]] += 1
            if rec["decision"] != "allow":
                reasons[rec["reason_code"]] += 1
            if rec["attempt"] > 0:
                regenerations += 1
    return {
        "sessions_with_audit": sessions,
        "decisions": dict(decisions),
        "normalizations": dict(normalizations),
        "rejection_reasons": dict(reasons),
        "regenerations": regenerations,
    }
