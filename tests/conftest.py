from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "models"


def _weights_present(model_id: str) -> bool:
    from mival.modelcard import load_card

    card = load_card(REGISTRY / f"{model_id}.json")
    return all(Path(w["uri"]).exists() for w in card.weights)


@pytest.fixture(scope="session")
def registry_dir() -> Path:
    return REGISTRY


# Several tests assert that no backend has leaked into ``sys.modules`` — the
# concrete form of the dual-environment rule (`mival` core must import in both
# the torch and the keras27 environment). Those assertions are in-process, so
# they only hold while no earlier test module has imported a backend. Keep any
# test module that does import one named ``test_torch_*`` or ``test_keras_*``,
# which collects after them, and import the backend inside functions rather
# than at module scope.


def pytest_runtest_setup(item):
    # The weights check is cheap and side-effect free, so it runs first: a
    # machine without weight files must skip before any backend is imported.
    marker = item.get_closest_marker("weights")
    if marker and marker.args:
        model_id = marker.args[0]
        if not _weights_present(model_id):
            pytest.skip(f"weights for {model_id} not present on this machine")
    if item.get_closest_marker("torch"):
        pytest.importorskip("torch")
    if item.get_closest_marker("keras"):
        pytest.importorskip("tensorflow")
