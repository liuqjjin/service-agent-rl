"""Audit trail for governance decisions.

Every candidate action gets a record -- allowed or not. Rejected candidates
exist ONLY here: they must never enter the official simulation trajectory,
because the evaluator replays trajectory tool calls against a fresh
environment (see UPSTREAM.md, "The evaluator replays trajectory writes").
The audit trail is also the data source for the governance metrics reported
in the ablation (candidate counts, denial reasons, regeneration counts).

task_id lives in the audit record for offline analysis. It is never shown to
the model; the leak tests police that boundary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from service_agent.governance.core import GovernanceResult, ProposedAction


@dataclass(frozen=True)
class AuditRecord:
    session_id: str
    task_id: str
    # 0 = first candidate for this turn; 1+ = retry candidate index. A
    # multi-tool candidate produces one record per proposed action, so records
    # with attempt > 0 are not themselves a count of regeneration rounds.
    attempt: int
    tool_name: str
    arguments_json: str
    decision: str
    reason_code: str
    policy_ref: str
    tool_call_id: str = ""


@dataclass
class AuditTrail:
    session_id: str
    task_id: str = ""
    records: list[AuditRecord] = field(default_factory=list)

    def record(
        self, candidate: ProposedAction, result: GovernanceResult, attempt: int
    ) -> AuditRecord:
        rec = AuditRecord(
            session_id=self.session_id,
            task_id=self.task_id,
            attempt=attempt,
            tool_name=candidate.tool_name,
            arguments_json=json.dumps(candidate.arguments, sort_keys=True),
            decision=result.decision.value,
            reason_code=result.reason_code,
            policy_ref=result.policy_ref,
            tool_call_id=candidate.tool_call_id,
        )
        self.records.append(rec)
        return rec

    # -- metrics ---------------------------------------------------------------

    def counts_by_decision(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.records:
            out[rec.decision] = out.get(rec.decision, 0) + 1
        return out

    def rejected_candidates(self) -> list[AuditRecord]:
        return [r for r in self.records if r.decision != "allow"]

    def retry_decision_record_count(self) -> int:
        """Count adjudicated action records emitted by retry candidates."""
        return sum(1 for r in self.records if r.attempt > 0)

    # -- persistence -----------------------------------------------------------

    def dump_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # One path belongs to one session and is a snapshot of this in-memory
        # trail. stop() may be called more than once during cleanup; replacing
        # the snapshot keeps persistence idempotent instead of duplicating every
        # record on the second call.
        with path.open("w") as f:
            for rec in self.records:
                f.write(json.dumps(asdict(rec)) + "\n")
