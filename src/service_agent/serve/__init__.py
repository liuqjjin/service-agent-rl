"""HTTP serving layer: the shim that lets ART's tau-bench client drive tau2."""

from service_agent.serve.tau2_shim import create_app, main

__all__ = ["create_app", "main"]
