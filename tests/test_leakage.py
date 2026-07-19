"""Leak detection must (a) pass on the prompts we actually build, and
(b) actually fire on the known-dangerous serializations. A leak test that
cannot detect a planted leak proves nothing."""

import pytest

from service_agent.leakage import LabelLeakError, assert_no_leak, find_leaks
from service_agent.splits import load_frozen_dev_ids

# A governance-relevant task (payment + resume rules) plus a long fault chain.
SAMPLE_IDS = [
    "[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:None]",
    "[service_issue]contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]",
]


@pytest.fixture(scope="module")
def sample_tasks(telecom_tasks):
    return [telecom_tasks[tid] for tid in SAMPLE_IDS]


def test_detector_fires_on_str_task(sample_tasks):
    # str(task) is the exact accident the gym info dict invites
    # (gym_agent.py:728 returns the full Task; Task.__str__ prints
    # "Evaluation Criteria:" with the reference actions).
    for task in sample_tasks:
        assert "str(task)" in find_leaks(str(task), task)
        assert "Evaluation Criteria:" in str(task)
        # A partial dump of just the criteria (indented differently inside
        # str(task), hence a separate check) must also be caught.
        assert "str(task.evaluation_criteria)" in find_leaks(
            str(task.evaluation_criteria), task
        )


def test_detector_fires_on_task_id_segment(sample_tasks):
    task = sample_tasks[0]
    text = "I suspect the problem is overdue_bill_suspension on this line."
    assert "task.id.segment" in find_leaks(text, task)


def test_detector_fires_on_criteria_json(sample_tasks):
    import json

    for task in sample_tasks:
        blob = json.dumps(
            task.evaluation_criteria.model_dump(exclude_none=True), ensure_ascii=False
        )
        assert find_leaks(blob, task), "criteria JSON dump must be detected"


def test_lax_mode_permits_user_volunteered_info(telecom_tasks):
    # Mid-conversation, users legitimately state their own name and number
    # (known_info). Lax mode must not flag that, but must still flag the
    # answer key.
    task = telecom_tasks[SAMPLE_IDS[0]]
    instructions = task.user_scenario.instructions
    if instructions.known_info:
        assert find_leaks(instructions.known_info, task, strict=False) == []
    assert find_leaks(str(task.evaluation_criteria), task, strict=False)


def test_native_agent_system_prompt_is_clean(telecom_tasks, telecom_env_constructor):
    # The prompt surface H0 actually uses: policy + tool schemas. Must contain
    # nothing task-derived, for every dev task.
    from tau2.agent.llm_agent import LLMAgent

    env = telecom_env_constructor()
    agent = LLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="mock/none",
    )
    prompt = agent.system_prompt
    for tid in load_frozen_dev_ids():
        assert_no_leak(prompt, telecom_tasks[tid], context=f"system prompt vs {tid}")


def test_assert_no_leak_raises(sample_tasks):
    task = sample_tasks[0]
    with pytest.raises(LabelLeakError):
        assert_no_leak(str(task), task)
