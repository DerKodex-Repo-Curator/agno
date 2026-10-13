"""Ownership-scoped checkpoint inspection and downloads through real AgentOS."""

import asyncio

import pytest
from sqlalchemy import update

from agno.sandbox.artifacts import SandboxArtifactNotFound, SandboxArtifacts, SandboxSessionNotFound
from agno.sandbox.base import SandboxError
from agno.sandbox.checkpoint import CheckpointRef
from tests.integration.sandbox.test_agentos import IMAGE, REDIS, URL
from tests.integration.sandbox.test_agentos import service as _service_fixture

service = _service_fixture

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not URL or not IMAGE or not REDIS, reason="Enable dedicated local sandbox services"),
]


async def completed(service):
    client, agent, _, _ = service
    response = await client.post(
        "/agents/coder/runs",
        data={"message": "artifact", "session_id": "session", "stream": "false", "user_id": "owner"},
    )
    assert response.status_code == 200, response.text
    return response.json()["run_id"], SandboxArtifacts(agent.coordinator.store, agent.coordinator.transfer.store)


async def test_download_after_compute_is_destroyed(service):
    client, agent, _, handles = service
    run_id, artifacts = await completed(service)
    assert await agent.coordinator.provider.aget(handles[0].provider_ref) is None
    state = await artifacts.astatus(agent_id="coder", session_id="session", user_id="owner")
    assert state["phase"] == "idle" and state["has_checkpoint"] and not state["compute_allocated"]
    listing = await client.get("/agents/coder/sessions/session/sandbox/artifacts")
    assert listing.status_code == 200 and listing.json()["run_id"] == run_id, listing.text
    assert any(value["path"] == "fixture-effects.txt" for value in listing.json()["files"])
    response = await client.get(
        "/agents/coder/sessions/session/sandbox/artifact", params={"path": "fixture-effects.txt"}
    )
    assert response.status_code == 200 and response.text == run_id + "\n", response.text
    assert response.headers["x-agno-checkpoint"] and response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"].startswith("attachment;")
    # Pins the runtime media type against the route's OpenAPI declaration.
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["cache-control"] == "private, no-store"
    sync = await asyncio.to_thread(
        artifacts.read, "fixture-effects.txt", agent_id="coder", session_id="session", user_id="owner"
    )
    assert sync.content == response.content


async def test_exact_owner_and_agent_required(service):
    _, artifacts = await completed(service)
    for agent_id, user_id in (("coder", None), ("coder", "attacker"), ("different", "owner")):
        with pytest.raises(SandboxSessionNotFound):
            await artifacts.aread("fixture-effects.txt", agent_id=agent_id, session_id="session", user_id=user_id)
    client, _, _, _ = service
    assert (await client.get("/agents/other/sessions/session/sandbox")).status_code == 404


async def test_paths_native_state_and_limits(service):
    _, artifacts = await completed(service)
    for path in ("../native/session.db", "/runtime/native/session.db", "a/../fixture-effects.txt", "a\\b", ""):
        with pytest.raises(SandboxError):
            await artifacts.aread(path, agent_id="coder", session_id="session", user_id="owner")
    with pytest.raises(SandboxArtifactNotFound):
        await artifacts.aread("native/session.db", agent_id="coder", session_id="session", user_id="owner")
    artifacts.max_file_bytes = 1
    with pytest.raises(SandboxError, match="limit"):
        await artifacts.aread("fixture-effects.txt", agent_id="coder", session_id="session", user_id="owner")


async def test_replaced_session_cannot_read_previous_incarnation(service):
    _, artifacts = await completed(service)
    store = artifacts.store
    await store._transaction(lambda c: c.execute(update(store.sessions).values(session_data={})))
    with pytest.raises(SandboxSessionNotFound):
        await artifacts.alist(agent_id="coder", session_id="session", user_id="owner")


async def test_checkpoint_identity_and_bytes_are_verified(service):
    _, artifacts = await completed(service)
    record = await artifacts._record("coder", "session", "owner")
    wrong = dict(record)
    wrong["checkpoint_identity"] = {**record["checkpoint_identity"], "session_id": "other"}
    with pytest.raises(SandboxError, match="match"):
        await asyncio.to_thread(artifacts._inspect, wrong, "fixture-effects.txt", 0, 1)
    (artifacts.objects.root / record["checkpoint"]["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(SandboxError, match="size"):
        await artifacts.alist(agent_id="coder", session_id="session", user_id="owner")


async def test_http_user_isolation_and_missing_identity(service):
    import httpx

    await completed(service)
    _, _, app, _ = service

    # Exercise the endpoint against identity already verified by auth middleware;
    # JWT signature verification belongs to the existing middleware test suite.
    async def scoped(scope, receive, send):
        scope.setdefault("state", {}).update(user_id="attacker", user_isolation_enabled=True)
        await app(scope, receive, send)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=scoped), base_url="http://test") as client:
        response = await client.get(
            "/agents/coder/sessions/session/sandbox/artifact",
            params={"path": "fixture-effects.txt", "user_id": "owner"},
        )
        assert response.status_code == 404, response.text


async def test_artifacts_remain_at_last_completed_turn_during_new_execution(service):
    client, agent, app, _ = service
    first, artifacts = await completed(service)
    agent.runtime_config = {"delay": 10}
    response = await client.post(
        "/agents/coder/runs",
        data={
            "message": "pending",
            "session_id": "session",
            "user_id": "owner",
            "stream": "false",
            "background": "true",
        },
    )
    run_id = response.json()["run_id"]

    async def running():
        while (await artifacts.astatus(agent_id="coder", session_id="session", user_id="owner"))["phase"] == "idle":
            await asyncio.sleep(0.05)

    await asyncio.wait_for(running(), 10)
    result = await artifacts.aread("fixture-effects.txt", agent_id="coder", session_id="session", user_id="owner")
    assert result.run_id == first and result.content == (first + "\n").encode()
    # The same selection through the registered route: the live workspace has
    # already been changed by the running turn, so different bytes exist.
    served = await client.get("/agents/coder/sessions/session/sandbox/artifact", params={"path": "fixture-effects.txt"})
    assert served.status_code == 200, served.text
    assert served.content == (first + "\n").encode()
    assert served.headers["x-agno-run-id"] == first
    await client.post(f"/agents/coder/runs/{run_id}/cancel", params={"session_id": "session"})


async def test_malformed_artifact_paths_are_client_errors(service):
    client, _, _, _ = service
    await completed(service)
    for path in (
        "../native/session.db",
        "/runtime/native/session.db",
        "a/../fixture-effects.txt",
        "a\\b",
        "",
        "fixture-effects.txt/",
    ):
        response = await client.get("/agents/coder/sessions/session/sandbox/artifact", params={"path": path})
        assert response.status_code == 400, (path, response.status_code, response.text)


async def test_addressable_artifact_paths_with_no_content_are_not_found(service):
    client, _, _, _ = service
    await completed(service)
    # Native state is unaddressable rather than forbidden: the workspace prefix
    # makes it a path the archive does not contain.
    for path in ("native/session.db", "absent.txt"):
        response = await client.get("/agents/coder/sessions/session/sandbox/artifact", params={"path": path})
        assert response.status_code == 404, (path, response.status_code, response.text)


async def test_resolution_precedes_path_validation(service):
    client, _, _, _ = service
    await completed(service)
    response = await client.get(
        "/agents/coder/sessions/absent-session/sandbox/artifact", params={"path": "../native/session.db"}
    )
    assert response.status_code == 404, response.text


async def test_omitted_artifact_path_stays_a_validation_error(service):
    client, _, _, _ = service
    await completed(service)
    response = await client.get("/agents/coder/sessions/session/sandbox/artifact")
    assert response.status_code == 422, response.text


class _Recorded:
    """Delegates to the real object, recording which attributes were used."""

    def __init__(self, target, log, label):
        self._target, self._log, self._label = target, log, label

    def __getattr__(self, name):
        value = getattr(self._target, name)
        if callable(value):

            def recorded(*args, **kwargs):
                self._log.append(self._label + "." + name + "()")
                return value(*args, **kwargs)

            return recorded
        self._log.append(self._label + "." + name)
        return value


async def test_reads_never_reach_the_compute_provider(service):
    """Status, listing and download are persisted-state reads.

    Warm reuse makes this load-bearing: a read must not create, resume, execute
    on or otherwise mutate compute, so the provider must not be called at all.
    """
    client, agent, _, _ = service
    await completed(service)
    log: list = []
    coordinator = agent.coordinator
    provider, transfer = coordinator.provider, coordinator.transfer
    coordinator.provider = _Recorded(provider, log, "provider")
    coordinator.transfer = _Recorded(transfer, log, "transfer")
    try:
        status = await client.get("/agents/coder/sessions/session/sandbox")
        listing = await client.get("/agents/coder/sessions/session/sandbox/artifacts")
        download = await client.get(
            "/agents/coder/sessions/session/sandbox/artifact", params={"path": "fixture-effects.txt"}
        )
    finally:
        coordinator.provider, coordinator.transfer = provider, transfer
    assert (status.status_code, listing.status_code, download.status_code) == (200, 200, 200)
    assert [entry for entry in log if entry.endswith("()")] == [], log
    assert [entry for entry in log if entry.startswith("provider")] == [], log


async def test_a_reference_whose_object_is_gone_is_storage_unavailable(service):
    """Retention or storage loss leaves a published reference with no object.

    That is storage being unavailable, not a servable conflict, so it must not be
    answered as a 409 by the generic SandboxError clause.
    """
    client, _, _, _ = service
    _, artifacts = await completed(service)
    record = await artifacts._record("coder", "session", "owner")
    await artifacts.objects.adelete(CheckpointRef(**record["checkpoint"]))

    download = await client.get(
        "/agents/coder/sessions/session/sandbox/artifact", params={"path": "fixture-effects.txt"}
    )
    assert download.status_code == 503, (download.status_code, download.text)
    assert download.json()["detail"] == "Checkpoint storage is unavailable"
    listing = await client.get("/agents/coder/sessions/session/sandbox/artifacts")
    assert listing.status_code == 503, (listing.status_code, listing.text)
    # Status reads the persisted record only, so it still answers and still reports
    # the checkpoint the session committed.
    status = await client.get("/agents/coder/sessions/session/sandbox")
    assert status.status_code == 200 and status.json()["has_checkpoint"] is True
    # A malformed client path is still a client error in this state.
    malformed = await client.get("/agents/coder/sessions/session/sandbox/artifact", params={"path": "../escape"})
    assert malformed.status_code == 400, malformed.text
