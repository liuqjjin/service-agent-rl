"""Keep vLLM's CUDA allocator from selecting TileLang's linker stub.

ART's isolated runtime imports DeepSeek-specific TileLang patches before vLLM
initializes its sleep-mode allocator. TileLang loads ``libcudart_stub.so`` into
the process, and vLLM otherwise mistakes the first matching memory map for the
real CUDA runtime. The training driver sets ``VLLM_CUDART_SO_PATH`` only after
verifying the runtime-owned library and its ``cudaDeviceReset`` symbol.
"""

from __future__ import annotations

import os


def _install_cudart_override() -> None:
    cudart_path = os.environ.get("VLLM_CUDART_SO_PATH")
    if not cudart_path:
        return

    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":
            return
        raise

    from vllm.utils import system_utils

    original = system_utils.find_loaded_library
    if getattr(original, "__service_agent_cudart_override__", False):
        return

    def find_loaded_library(name: str) -> str | None:
        if name == "libcudart":
            return cudart_path
        return original(name)

    find_loaded_library.__service_agent_cudart_override__ = True  # type: ignore[attr-defined]
    system_utils.find_loaded_library = find_loaded_library


_install_cudart_override()
