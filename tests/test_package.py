"""Smoke test: the package is installed and importable."""

import mlops_live


def test_package_has_version() -> None:
    assert isinstance(mlops_live.__version__, str)
    assert mlops_live.__version__ == "0.1.0"
