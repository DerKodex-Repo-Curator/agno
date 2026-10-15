"""Projected run-preview reads against a live MongoDB (see the unit twin for the corpus)."""

import os
import uuid
from typing import Any, Dict, List

import pytest

from tests.unit.db.test_session_run_previews import check_backend, check_projected_query

pymongo = pytest.importorskip("pymongo")
from pymongo import monitoring  # noqa: E402

MONGO_URL = os.getenv("AGNO_TEST_MONGO_URL", "mongodb://localhost:27017")

pytestmark = pytest.mark.integration


class _RunsReads(monitoring.CommandListener):
    """Captures reads on the runs collection to prove ``run_data`` is never returned whole."""

    def __init__(self) -> None:
        self.commands: List[Dict[str, Any]] = []

    def started(self, event: Any) -> None:
        if event.command_name in ("find", "aggregate") and "runs" in str(event.command.get(event.command_name)):
            self.commands.append(dict(event.command))

    def succeeded(self, event: Any) -> None:
        pass

    def failed(self, event: Any) -> None:
        pass

    def assert_projected(self) -> None:
        assert self.commands, "expected the projected read to hit the runs collection"
        for command in self.commands:
            if "find" in command:
                projection = command.get("projection") or {}
                assert projection and "run_data" not in projection, command
            else:
                stages = [next(iter(stage)) for stage in command["pipeline"]]
                assert stages[-1] == "$project", command
                assert "run_data" not in command["pipeline"][-1]["$project"], command


@pytest.fixture
def db_name():
    client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("MongoDB not reachable")
    name = f"previews_{uuid.uuid4().hex[:8]}"
    yield name
    client.drop_database(name)
    client.close()


async def test_mongo_previews_match_full_path(db_name):
    from agno.db.mongo import MongoDb

    await check_backend(MongoDb(db_url=MONGO_URL, db_name=db_name))


async def test_mongo_previews_are_projected(monkeypatch, db_name):
    from agno.db.mongo import MongoDb

    listener = _RunsReads()
    client = pymongo.MongoClient(MONGO_URL, event_listeners=[listener])
    await check_projected_query(monkeypatch, MongoDb(db_client=client, db_name=db_name))
    listener.assert_projected()
    client.close()


async def test_async_mongo_previews_match_full_path(db_name):
    from agno.db.mongo import AsyncMongoDb

    await check_backend(AsyncMongoDb(db_url=MONGO_URL, db_name=db_name))


async def test_async_mongo_previews_are_projected(monkeypatch, db_name):
    from pymongo import AsyncMongoClient

    from agno.db.mongo import AsyncMongoDb

    listener = _RunsReads()
    client: Any = AsyncMongoClient(MONGO_URL, event_listeners=[listener])
    await check_projected_query(monkeypatch, AsyncMongoDb(db_client=client, db_name=db_name))
    listener.assert_projected()
    await client.close()
