"""
ASTRA pytest fixtures shared across the test suite.
"""

import os
import random
import string
import sys

import pytest

# Ensure the backend root is on sys.path so `import app...` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session", autouse=True)
def _force_paper_trading():
    """Safety net: every test runs with paper trading hard-locked."""
    os.environ["PAPER_TRADING"] = "true"
    yield


@pytest.fixture()
def random_username() -> str:
    return "t_" + "".join(random.choices(string.ascii_lowercase, k=10))


@pytest.fixture()
def jwt_client(monkeypatch):
    """TestClient with AUTH_MODE=jwt forced. Reloads endpoints module to re-evaluate guard."""
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("PAPER_TRADING", "true")
    import importlib
    import app.api.endpoints as ep_mod
    importlib.reload(ep_mod)
    import main as main_mod
    importlib.reload(main_mod)
    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


@pytest.fixture()
def bypass_client(monkeypatch):
    """TestClient with AUTH_MODE=bypass — used to test the bypass path explicitly."""
    monkeypatch.setenv("AUTH_MODE", "bypass")
    monkeypatch.setenv("PAPER_TRADING", "true")
    import importlib
    import app.api.endpoints as ep_mod
    importlib.reload(ep_mod)
    import main as main_mod
    importlib.reload(main_mod)
    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)
