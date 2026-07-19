import pytest


@pytest.fixture(scope="session")
def telecom_tasks():
    """All telecom tasks as typed Task objects, keyed by id. Heavy import, do once."""
    from tau2.registry import registry

    tasks = registry.get_tasks_loader("telecom")()
    return {t.id: t for t in tasks}


@pytest.fixture(scope="session")
def telecom_env_constructor():
    from tau2.registry import registry

    return registry.get_env_constructor("telecom")
