"""Download and verify the one checkpoint allowed by the protocol."""

from __future__ import annotations

import os
from pathlib import Path

from service_agent.training.contracts import BASE_MODEL_ID, BASE_MODEL_REVISION


def prepare_pinned_snapshot(cache_dir: Path) -> Path:
    """Populate a dedicated HF cache and prove that `main` is still the pin.

    ART's training tokenizer currently loads the model ID without forwarding
    `revision`. Downloading `main` only after checking its SHA, then validating
    the returned snapshot directory, makes that no-revision load resolve to the
    same files as the explicitly pinned trainer and vLLM loaders.
    """

    from huggingface_hub import HfApi, snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    actual = HfApi().model_info(BASE_MODEL_ID, revision="main").sha
    if actual != BASE_MODEL_REVISION:
        raise RuntimeError(
            f"{BASE_MODEL_ID} main moved: expected {BASE_MODEL_REVISION}, got {actual}"
        )
    snapshot = Path(
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            revision="main",
            cache_dir=cache_dir,
        )
    ).resolve()
    if snapshot.name != BASE_MODEL_REVISION:
        raise RuntimeError(
            "downloaded snapshot does not match the pinned revision: "
            f"{snapshot.name} != {BASE_MODEL_REVISION}"
        )
    # ART's pinned training-tokenizer helper does not forward `revision`.
    # `snapshot_download(..., revision="main")` wrote a verified refs/main
    # entry; forcing all later loaders offline prevents that helper from
    # resolving a newer moving main between preflight and an update.
    os.environ["HF_HUB_OFFLINE"] = "1"
    return snapshot
