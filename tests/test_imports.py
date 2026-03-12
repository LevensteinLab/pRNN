import importlib.util
import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        "prnn.utils.predictiveNet",
        "prnn.analysis.trajectoryAnalysis",
    ],
)
def test_import(package_name):
    spec = importlib.util.find_spec(package_name)
    assert spec is not None, f"Failed to import {package_name}"
