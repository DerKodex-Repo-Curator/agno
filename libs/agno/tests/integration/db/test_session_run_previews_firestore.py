"""Projected run-preview reads against the Firestore emulator (see the unit twin for the corpus).

Run the emulator and point FIRESTORE_EMULATOR_HOST at it, e.g.
``gcloud emulators firestore start --host-port=localhost:8080``.
"""

import os
import uuid

import pytest

from tests.unit.db.test_session_run_previews import check_backend, check_projected_query

pytest.importorskip("google.cloud.firestore")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("FIRESTORE_EMULATOR_HOST"), reason="FIRESTORE_EMULATOR_HOST not set"),
]


@pytest.fixture
def firestore_db():
    from google.cloud.firestore import Client

    from agno.db.firestore import FirestoreDb

    suffix = uuid.uuid4().hex[:8]
    return FirestoreDb(
        db_client=Client(project="agno-test"),
        session_collection=f"previews_sessions_{suffix}",
        runs_collection=f"previews_runs_{suffix}",
    )


async def test_firestore_previews_match_full_path(firestore_db):
    await check_backend(firestore_db)


async def test_firestore_previews_are_projected(monkeypatch, firestore_db):
    await check_projected_query(monkeypatch, firestore_db)
