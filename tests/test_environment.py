"""SETUP-01: the active venv matches the exact pins in requirements.txt (pickles are version-sensitive)."""
from importlib.metadata import version

import pytest
from conftest import ROOT

pytestmark = pytest.mark.unit


def _pins():
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and "==" in line:
            name, ver = line.split("==")
            yield name.strip(), ver.strip()


@pytest.mark.parametrize("name,ver", list(_pins()))
def test_installed_version_matches_pin(name, ver):
    assert version(name) == ver
