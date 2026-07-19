"""GovernedLLMAgent: adjudicate candidate actions before they exist officially.

Placement is the whole point. The evaluator computes the DB reward by
replaying every tool call found in the official trajectory against a fresh
environment (UPSTREAM.md, "The evaluator replays trajectory writes"), and
set_state additionally requires each trajectory tool call to be followed by
its matching ToolMessage. A rejected action must therefore never reach the
trajectory at all -- not "reach it but be skipped at execution", which would
either corrupt the replayed state or crash the replay outright.

The only place that satisfies this is inside the agent, before the message is
returned to the Orchestrator: whatever generate_next_message returns is what
the Orchestrator appends to the trajectory (orchestrator.py:862-870). So this
agent generates a candidate, adjudicates it, and only returns it if allowed.
Rejected candidates are recorded in the audit trail and turned into a private
SystemMessage appended to the next generation's input -- never into the
agent state, so never into the trajectory and never into the user simulator's
view. Regeneration is bounded; when the budget runs out the agent sends a
safe text message instead of an unauthorized write.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from service_agent.governance.audit import AuditTrail
from service_agent.governance.core import (
    Decision,
    GovernanceResult,
    ProposedAction,
    idempotency_key,
)
from service_agent.governance.evidence import EvidenceExtractor
from service_agent.governance.recovery import RecoveryBudget, classify_tool_error
from service_agent.governance.telecom_rules import WRITE_TOOLS, TelecomGovernor

FEEDBACK_PREFIX = "[execution governance — internal, do not mention to the customer]"

FALLBACK_TEXT = (
    "I want to make sure I have everything verified correctly before making "
    "that change. Could you bear with me while I double-check the details, "
    "or let me know if there is anything else I can help you with?"
)


@dataclass(frozen=True)
class GovernanceConfig:
    """H1 = gate only. H2 = gate + idempotency + bounded recovery."""

    max_regenerations: int = 2
    enable_idempotency: bool = True
    enable_recovery: bool = True
    max_recoveries: int = 4

    @classmethod
    def h1(cls) -> "GovernanceConfig":
        return cls(enable_idempotency=False, enable_recovery=False)

    @classmethod
    def h2(cls) -> "GovernanceConfig":
        return cls()


class GovernedLLMAgent(LLMAgent):
    """LLMAgent plus a deterministic execution gate at the generation boundary."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        config: Optional[GovernanceConfig] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.config = config or GovernanceConfig()
        self.session_id = uuid.uuid4().hex
        self.governor = TelecomGovernor(session_id=self.session_id)
        self.extractor = EvidenceExtractor()
        self.audit = AuditTrail(session_id=self.session_id)
        self.recovery = RecoveryBudget(max_recoveries=self.config.max_recoveries)
        # Set by the eval harness; when present, stop() persists the audit
        # trail there so batch runs keep per-simulation governance records.
        self.audit_dir = None

    def stop(self, message=None, state=None) -> None:
        super().stop(message, state)
        if self.audit_dir is not None and self.audit.records:
            from pathlib import Path

            self.audit.dump_jsonl(
                Path(self.audit_dir) / f"audit_{self.session_id}.jsonl"
            )

    # -- state ------------------------------------------------------------------

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        state = super().get_init_state(message_history)
        # Tasks may start mid-conversation; the gate's evidence must include
        # what that history already established.
        for msg in state.messages:
            self._observe(msg)
        return state

    # -- generation with adjudication -------------------------------------------

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio.")
        if isinstance(message, MultiToolMessage):
            for tool_msg in message.tool_messages:
                self._observe(tool_msg)
            state.messages.extend(message.tool_messages)
        else:
            self._observe(message)
            state.messages.append(message)

        # Rejected-candidate feedback is folded into an amended copy of the
        # system prompt rather than appended as a trailing system message:
        # serving stacks (mlx_lm.server, and most Qwen chat templates) only
        # accept system content at position zero. The amendment is ephemeral --
        # state.system_messages is never modified, so nothing reaches the
        # official trajectory.
        feedback_notes: list[str] = []
        verdict: Optional[GovernanceResult] = None
        for attempt in range(self.config.max_regenerations + 1):
            system_messages = state.system_messages
            if feedback_notes:
                amended = SystemMessage(
                    role="system",
                    content=state.system_messages[0].content
                    + "\n\n"
                    + "\n\n".join(feedback_notes),
                )
                system_messages = [amended] + state.system_messages[1:]
            candidate = generate(
                model=self.llm,
                tools=self.tools,
                messages=system_messages + state.messages,
                call_name="governed_agent_response",
                **self.llm_args,
            )
            self._sanitize_mixed(candidate, attempt)
            verdict = self._adjudicate(candidate)
            if candidate.is_tool_call() or not verdict.allowed or attempt > 0:
                for proposed in self._proposed_actions(candidate):
                    self.audit.record(proposed, verdict, attempt)
            if verdict.allowed:
                self._observe(candidate)
                return candidate
            feedback_notes.append(self._feedback_note(candidate, verdict))

        # Regeneration budget exhausted: never return the rejected candidate.
        fallback = AssistantMessage(role="assistant", content=FALLBACK_TEXT, cost=0.0)
        self._observe(fallback)
        return fallback

    def _sanitize_mixed(self, candidate: AssistantMessage, attempt: int) -> None:
        """Mixed text + tool call: strip the text, keep the call.

        The policy says one or the other (main_policy.md:9), but this is a
        formatting habit, not a business violation: the orchestrator routes a
        mixed message to the environment and the user never sees its text, and
        telecom's reward basis carries no COMMUNICATE component. Qwen3.5-4B
        mixes roughly four times per episode; denying and regenerating burned
        turns to max-steps in the smoke run (103 denials over 3 episodes), so
        the gate normalizes instead of fighting the model's formatting. The
        audit keeps a record; the harness ablation reports the rate.
        """
        if candidate.is_tool_call() and candidate.has_text_content():
            self.audit.record(
                ProposedAction.from_tool_call(candidate.tool_calls[0]),
                GovernanceResult(
                    Decision.ALLOW,
                    "mixed_text_stripped",
                    "",
                    policy_ref="main_policy.md:9",
                ),
                attempt,
            )
            candidate.content = None

    def _adjudicate(self, candidate: AssistantMessage) -> GovernanceResult:
        if not candidate.is_tool_call():
            return GovernanceResult(Decision.ALLOW, "text_message", "")
        if len(candidate.tool_calls) > 1:
            return GovernanceResult(
                Decision.DENY,
                "multiple_tool_calls",
                "Make only one tool call at a time.",
                policy_ref="main_policy.md:9",
            )

        proposed = ProposedAction.from_tool_call(candidate.tool_calls[0])
        if (
            self.config.enable_recovery
            and self.recovery.exhausted
            and proposed.tool_name in WRITE_TOOLS
        ):
            return GovernanceResult(
                Decision.TRANSFER,
                "recovery_budget_exhausted",
                "Too many failed attempts in this conversation. Stop retrying "
                "writes; transfer the customer to a human agent "
                "(transfer_to_human_agents) or end safely.",
                policy_ref="main_policy.md:13",
            )
        verdict = self.governor.evaluate(proposed, self.extractor.state)
        if verdict.decision is Decision.DUPLICATE and not self.config.enable_idempotency:
            # H1 has no idempotency ledger; the duplicate goes through and the
            # ablation measures the resulting double side effects.
            return GovernanceResult(Decision.ALLOW, "duplicate_allowed_h1", "")
        return verdict

    # -- evidence plumbing -------------------------------------------------------

    def _observe(self, message: Message) -> None:
        if isinstance(message, AssistantMessage):
            self.extractor.observe_agent_message(message)
        elif isinstance(message, UserMessage):
            self.extractor.observe_user_message(message)
        elif isinstance(message, ToolMessage):
            if message.error:
                if self.config.enable_recovery:
                    self.recovery.spend(classify_tool_error(message.content or ""))
                self.extractor.observe_tool_result(message)
                return
            key = ""
            call = self.extractor.state.pending_calls.get(message.id)
            if call is not None and call[0] in WRITE_TOOLS:
                key = idempotency_key(self.session_id, call[0], call[1])
            self.extractor.observe_tool_result(message, idempotency_key=key)

    @staticmethod
    def _proposed_actions(candidate: AssistantMessage) -> list[ProposedAction]:
        if not candidate.is_tool_call():
            return [ProposedAction(tool_name="<text>", arguments={}, tool_call_id="")]
        return [ProposedAction.from_tool_call(tc) for tc in candidate.tool_calls]

    @staticmethod
    def _feedback_note(candidate: AssistantMessage, verdict: GovernanceResult) -> str:
        if candidate.is_tool_call():
            calls = ", ".join(
                f"{tc.name}({tc.arguments})" for tc in candidate.tool_calls
            )
            attempted = f"Your proposed call {calls} was not executed"
        else:
            attempted = "Your previous response was not sent"
        ref = f" [policy: {verdict.policy_ref}]" if verdict.policy_ref else ""
        return (
            f"{FEEDBACK_PREFIX} {attempted} "
            f"({verdict.reason_code}).{ref} {verdict.guidance} "
            "Respond again taking this into account."
        )


def create_governed_llm_agent(tools, domain_policy, **kwargs):
    """Factory with the signature tau2's registry expects."""
    return GovernedLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        config=kwargs.get("config"),
    )
