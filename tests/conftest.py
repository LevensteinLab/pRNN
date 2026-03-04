# tests/conftest.py
import pytest
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_nets():
    """Delete nets/tmp/ before the entire test session completes if there are old ones."""
    tmp_nets_dir = REPO_ROOT / "nets" / "tmp"
    if tmp_nets_dir.exists():
        shutil.rmtree(tmp_nets_dir)
        print(f"\nCleaned up {tmp_nets_dir}")

    """Delete nets/tmp/ after the entire test session completes."""
    yield  # all tests run here

   
    if tmp_nets_dir.exists():
        shutil.rmtree(tmp_nets_dir)
        print(f"\nCleaned up {tmp_nets_dir}")

