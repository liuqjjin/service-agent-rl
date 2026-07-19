"""Prepare SFT warm-start data from teacher runs (the GRPO fallback bridge).

If direct dual-control GRPO stalls (all-zero groups, flat reward), the fallback
is: run a strong teacher through the NATIVE runner on train-core, keep only
episodes that (a) earned official reward 1.0 and (b) replay clean through the
governor offline (no unauthorized writes -- we do not teach the student to
succeed by violating policy), and fine-tune the 4B on them before GRPO.

Teacher generation is just the ablation runner pointed at the teacher model:

    uv run python -m service_agent.eval.run_ablation --arm h0 --tasks train-core \
        --trials 2 --agent-llm "openai//Users/lqj/local-llm/models/Qwen3.6-35B-A3B-4bit" \
        --agent-api-base http://127.0.0.1:8399/v1 --out results/teacher/r1

This module filters and exports those results into chat-format JSONL, one
episode per line: {"messages": [...], "tools": [...], "task_id": ...}. The
sample contains exactly what the student would have seen -- the rebuilt
system prompt, user text, its own tool calls and their results. User-side
device-tool traffic is excluded (the agent never sees it live), and every
sample passes the label-leak check against its own task before export.
GPU-side, runbooks/autodl.md converts these lines into art.Trajectory
objects for model.train_sft.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results
from tau2.registry import registry

from service_agent.eval.metrics import analyze_trajectory
from service_agent.leakage import assert_no_leak
from service_agent.splits import load_split_ids, train_core_ids


def to_chat(messages) -> list[dict]:
    chat: list[dict] = []
    for msg in messages:
        if not is_valid_agent_history_message(msg):
            continue
        if isinstance(msg, UserMessage):
            chat.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantMessage):
            if msg.is_tool_call():
                chat.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                chat.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            chat.append(
                {"role": "tool", "tool_call_id": msg.id, "content": msg.content}
            )
    return chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env = registry.get_env_constructor("telecom")()
    system_prompt = LLMAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), llm="export"
    ).system_prompt
    tools = [tool.openai_schema for tool in env.get_tools()]
    tasks = {t.id: t for t in registry.get_tasks_loader("telecom")()}
    allowed_ids = set(train_core_ids())
    test_ids = load_split_ids()["test"]

    kept = skipped_reward = skipped_governance = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for results_path in args.results:
            results = Results.load(results_path)
            for sim in results.simulations:
                assert sim.task_id not in test_ids, (
                    "teacher results contain official test tasks; refusing to export"
                )
                if sim.task_id not in allowed_ids:
                    continue
                reward = sim.reward_info.reward if sim.reward_info else 0.0
                if reward is None or reward < 1.0:
                    skipped_reward += 1
                    continue
                analysis = analyze_trajectory(sim.messages)
                if analysis.unauthorized_executed_writes or analysis.duplicate_side_effects:
                    skipped_governance += 1
                    continue
                chat = [{"role": "system", "content": system_prompt}] + to_chat(
                    sim.messages
                )
                sample = {"task_id": sim.task_id, "messages": chat, "tools": tools}
                assert_no_leak(
                    json.dumps(chat, ensure_ascii=False),
                    tasks[sim.task_id],
                    strict=False,
                    context=f"SFT sample {sim.task_id}",
                )
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1

    print(
        json.dumps(
            {
                "kept": kept,
                "skipped_low_reward": skipped_reward,
                "skipped_governance": skipped_governance,
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
