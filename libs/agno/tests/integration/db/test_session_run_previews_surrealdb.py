"""Projected run-preview reads against a live SurrealDB (see the unit twin for the corpus)."""

import os
import uuid

import pytest

from tests.unit.db.test_session_run_previews import check_backend, check_projected_query

pytest.importorskip("surrealdb")

SURREALDB_URL = os.getenv("AGNO_TEST_SURREALDB_URL", "ws://localhost:8000")
CREDS = {"username": "root", "password": "root"}

pytestmark = pytest.mark.integration


@pytest.fixture
def surreal_db():
    from agno.db.surrealdb import SurrealDb

    try:
        db = SurrealDb(None, SURREALDB_URL, CREDS, "previews", f"db_{uuid.uuid4().hex[:8]}")
        db.client  # noqa: B018 - connect eagerly so an absent server skips
    except Exception:
        pytest.skip("SurrealDB not reachable")
    return db


async def test_surrealdb_previews_match_full_path(surreal_db):
    await check_backend(surreal_db)


async def test_surrealdb_previews_are_projected(monkeypatch, surreal_db):
    queries = []
    original = surreal_db._query

    def recording_query(query, vars, record_type):
        queries.append(query)
        return original(query, vars, record_type)

    monkeypatch.setattr(surreal_db, "_query", recording_query)
    await check_projected_query(monkeypatch, surreal_db)
    run_reads = [q for q in queries if "run_data" in q]
    assert run_reads and all("SELECT *" not in q and "run_data," not in q for q in run_reads), run_reads
