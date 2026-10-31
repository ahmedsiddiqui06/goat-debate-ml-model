"""API tests against the real built data (skipped if data/ hasn't been built)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "nba" / "seasons.csv").exists(),
                                reason="run `python -m goat.sources.nba` first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from goat.api import app
    return TestClient(app)


def test_compare_lebron_pele(client):
    r = client.get("/compare", params={"a": "lebron", "b": "pele", "boot": 50})
    assert r.status_code == 200
    j = r.json()
    assert j["a"]["athlete"] == "LeBron James" and j["b"]["athlete"] == "Pelé"
    assert 0 <= j["p_a_beats_b"] <= 1


def test_rankings_and_presets(client):
    r = client.get("/rankings", params={"limit": 5, "preset": "comparable_only", "boot": 50})
    assert r.status_code == 200
    assert len(r.json()["rankings"]) == 5
    assert r.json()["weights"]["recognition"] == 0


def test_athlete_detail(client):
    j = client.get("/athletes/Michael Jordan", params={"boot": 50}).json()
    assert j["sport"] == "basketball" and len(j["seasons"]) >= 13


def test_errors(client):
    assert client.get("/athletes/nobody at all").status_code == 404
    assert client.get("/compare", params={"a": "james", "b": "pele"}).status_code == 400
    assert client.get("/rankings", params={"weights": "bogus:1"}).status_code == 400
