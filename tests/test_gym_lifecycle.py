"""Gym wrapper lifecycle: seed propagation and orchestrator-thread cleanup.

Both bugs exist at upstream cf71a80 and are fixed on our submodule branch
(see UPSTREAM.md). To reproduce the failures on the pristine pin:

    git -C third_party/tau2-bench checkout cf71a8070269883e38a365ffa85f78f46844c1f4
    uv run pytest tests/test_gym_lifecycle.py

Bug 1 (seed): AgentGymEnv.reset(seed=...) forwards the seed to gymnasium's
np_random only; _get_orchestrator() builds the Orchestrator without it, so
Orchestrator.initialize never calls set_seed on the user simulator. Every
"seeded" gym run is actually unseeded.

Bug 2 (threads): the orchestrator thread blocks forever inside
GymAgent.generate_next_message waiting for set_action(). reset() joins the
old thread with timeout=1.0, which cannot succeed for a permanently blocked
thread, then leaks it as a daemon and starts a new one. Every reset leaks a
thread pinning the whole old orchestrator object graph.

Solo mode is used so the orchestrator blocks on the externally-controlled
GymAgent before any LLM call -- these tests need no API keys.
"""

import threading
import time

import pytest

from tau2.gym.gym_agent import AgentGymEnv, UserGymEnv


def _drain(env, timeout: float = 5.0) -> None:
    """Ask the env to shut down its orchestrator thread and wait for it."""
    close = getattr(env, "close", None)
    if close is not None:
        close()
    deadline = time.monotonic() + timeout
    thread = env._orchestrator_thread
    while thread and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)


@pytest.fixture
def solo_env():
    env = AgentGymEnv(domain="mock", task_id="create_task_1", solo_mode=True)
    yield env
    _drain(env)


def test_reset_seed_reaches_orchestrator(solo_env):
    solo_env.reset(seed=12345)
    assert solo_env._orchestrator is not None
    assert solo_env._orchestrator.seed == 12345


def test_reset_without_seed_keeps_orchestrator_unseeded(solo_env):
    solo_env.reset()
    assert solo_env._orchestrator.seed is None


def test_user_gym_env_seed_reaches_agent_llm():
    env = UserGymEnv(domain="mock", task_id="create_task_1")
    try:
        env.reset(seed=777)
        assert env._orchestrator.seed == 777
        # The automated LLMAgent is the seeded party in UserGymEnv;
        # Orchestrator.initialize must have pushed the seed into its llm_args.
        assert env._orchestrator.agent.llm_args.get("seed") == 777
    finally:
        _drain(env)


def test_repeated_reset_does_not_leak_threads(solo_env):
    solo_env.reset()
    baseline = threading.active_count()
    for _ in range(4):
        solo_env.reset()
    # Old orchestrator threads must exit once a reset abandons them. Allow
    # one in-flight thread for the current simulation, none accumulated.
    assert threading.active_count() <= baseline, (
        f"thread count grew from {baseline} to {threading.active_count()}: "
        "orchestrator threads leak across resets"
    )


def test_user_gym_env_reset_does_not_leak_threads():
    env = UserGymEnv(domain="mock", task_id="create_task_1")
    try:
        env.reset()
        baseline = threading.active_count()
        for _ in range(4):
            env.reset()
        assert threading.active_count() <= baseline
    finally:
        _drain(env)


def test_close_terminates_orchestrator_thread(solo_env):
    solo_env.reset()
    thread = solo_env._orchestrator_thread
    assert thread is not None and thread.is_alive()
    solo_env.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "close() must terminate the orchestrator thread"
