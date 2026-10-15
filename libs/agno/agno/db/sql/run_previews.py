"""Projected session-run reads for the run-previews listing (SQLite, Postgres, MySQL).

``run_data`` holds a whole serialized run, so reading it costs the full row however
little the caller needs. These statements pull single keys out with the dialect's JSON
operators instead; SQLAlchemy compiles the same expressions per dialect. The backends
run the statements (sync or async) and hand the rows to the pure helpers below.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import JSON, Table, case, func, literal_column, select, type_coerce
from sqlalchemy.sql import ColumnElement, Select

from agno.db.base import SessionType
from agno.db.utils import (
    PREVIEW_FALLBACK_KEYS,
    RUN_PREVIEW_KEYS,
    SESSION_PREVIEW_KEYS,
    complete_preview_session,
    deserialize_session_json_fields,
    preview_run_needs_full_data,
    resolve_preview_session_type,
    run_from_preview_projection,
)


@dataclass(frozen=True)
class JsonAccess:
    """How a dialect reads keys out of ``run_data``.

    ``extract`` yields a JSON-typed value; ``present`` is SQL NULL exactly when the key
    is absent (SQLite's JSON-typed reads wrap in JSON_QUOTE, which turns NULL into 'null');
    ``build_object`` folds the projected values into one JSON object per row, so each row
    is decoded once instead of once per key.
    """

    extract: Callable[[Any, Tuple[str, ...]], ColumnElement]
    present: Callable[[Any, Tuple[str, ...]], ColumnElement]
    build_object: Callable[..., ColumnElement]


def _index(data: Any, path: Tuple[str, ...]) -> Any:
    return data[path[0]] if len(path) == 1 else data[path]


def _as_string(data: Any, path: Tuple[str, ...]) -> Any:
    return _index(data, path).as_string()


def _singlestore_extract(data: Any, path: Tuple[str, ...]) -> Any:
    return type_coerce(func.JSON_EXTRACT_JSON(data, *path), JSON)


# SQLAlchemy's JSON indexing compiles to JSON_EXTRACT on SQLite/MySQL and ``->`` / ``#>`` on Postgres.
SQLITE_JSON = JsonAccess(extract=_index, present=_as_string, build_object=func.json_object)
POSTGRES_JSON = JsonAccess(extract=_index, present=_as_string, build_object=func.jsonb_build_object)
MYSQL_JSON = JsonAccess(extract=_index, present=_as_string, build_object=func.JSON_OBJECT)
# SingleStore takes keys as arguments rather than MySQL's ``$.`` paths.
SINGLESTORE_JSON = JsonAccess(
    extract=_singlestore_extract,
    present=lambda data, path: func.JSON_EXTRACT_JSON(data, *path),
    build_object=func.JSON_BUILD_OBJECT,
)


def session_stmt(sessions_table: Table, session_id: str, user_id: Optional[str]) -> Select:
    """Scoping and type columns, plus the legacy ``runs`` blob where the column still exists."""
    columns = [sessions_table.c[key] for key in SESSION_PREVIEW_KEYS if key in sessions_table.c]
    if "runs" in sessions_table.c:
        columns.append(sessions_table.c.runs)
    stmt = select(*columns).where(sessions_table.c.session_id == session_id)
    if user_id is not None:
        stmt = stmt.where(sessions_table.c.user_id == user_id)
    return stmt


def runs_stmt(runs_table: Table, session_id: str, json_access: JsonAccess) -> Select:
    """One JSON object per run holding the preview keys, ``input_content`` and ``workflow_input``."""
    data, extract = runs_table.c.run_data, json_access.extract
    values: Dict[str, Any] = {key: extract(data, (key,)) for key in RUN_PREVIEW_KEYS}
    values["input_content"] = extract(data, ("input", "input_content"))
    # A workflow run previews its raw input, which is small; other runs carry media beside input_content.
    has_workflow = json_access.present(data, ("workflow_id",)).is_not(None)
    values["workflow_input"] = case((has_workflow, extract(data, ("input",))))
    # Keys are this module's constants, inlined so Postgres need not infer a bind type.
    pairs = [part for key, value in values.items() for part in (literal_column(f"'{key}'"), value)]
    return (
        select(type_coerce(json_access.build_object(*pairs), JSON).label("preview"))
        .where(runs_table.c.session_id == session_id)
        .order_by(runs_table.c.run_index.asc(), runs_table.c.created_at.asc(), runs_table.c.run_id.asc())
    )


def fallback_runs_stmt(runs_table: Table, run_ids: Sequence[str], json_access: JsonAccess) -> Select:
    """What the input preview falls back to, for the few runs that stored no usable input."""
    data, extract = runs_table.c.run_data, json_access.extract
    return select(runs_table.c.run_id, *(extract(data, (key,)).label(key) for key in PREVIEW_FALLBACK_KEYS)).where(
        runs_table.c.run_id.in_(run_ids)
    )


def session_from_row(row: Any) -> Tuple[Dict[str, Any], Any]:
    """Session preview dict and the raw legacy ``runs`` blob from a ``session_stmt`` row."""
    values = deserialize_session_json_fields(dict(row._mapping))
    legacy_runs = values.pop("runs", None)
    return {key: values.get(key) for key in SESSION_PREVIEW_KEYS}, legacy_runs


def runs_from_rows(rows: Iterable[Any], session_type: SessionType) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Projected runs and the ids that still need their full row."""
    runs = [run_from_preview_projection(row.preview) for row in rows]
    missing = [run["run_id"] for run in runs if preview_run_needs_full_data(run, session_type)]
    return runs, missing


def complete_session(
    session: Dict[str, Any],
    legacy_runs: Any,
    runs: List[Dict[str, Any]],
    fallback_rows: Iterable[Any],
    session_type: SessionType,
) -> Dict[str, Any]:
    fallbacks: Dict[str, Dict[str, Any]] = {}
    for row in fallback_rows:
        values = dict(row._mapping)
        fallbacks[values["run_id"]] = {key: values[key] for key in PREVIEW_FALLBACK_KEYS if values.get(key) is not None}
    return complete_preview_session(session, legacy_runs, runs, fallbacks, session_type)


def read_session_previews(
    sess: Any,
    sessions_table: Table,
    runs_table: Optional[Table],
    session_id: str,
    session_type: Optional[SessionType],
    user_id: Optional[str],
    json_access: JsonAccess,
) -> Optional[Dict[str, Any]]:
    """Run the projected reads on a sync SQLAlchemy session."""
    row = sess.execute(session_stmt(sessions_table, session_id, user_id)).fetchone()
    if row is None:
        return None
    session, legacy_runs = session_from_row(row)
    resolved = resolve_preview_session_type(session, session_type)
    runs: List[Dict[str, Any]] = []
    fallback_rows: List[Any] = []
    if runs_table is not None:
        runs, missing = runs_from_rows(
            sess.execute(runs_stmt(runs_table, session_id, json_access)).fetchall(), resolved
        )
        if missing:
            fallback_rows = list(sess.execute(fallback_runs_stmt(runs_table, missing, json_access)).fetchall())
    return complete_session(session, legacy_runs, runs, fallback_rows, resolved)


async def aread_session_previews(
    sess: Any,
    sessions_table: Table,
    runs_table: Optional[Table],
    session_id: str,
    session_type: Optional[SessionType],
    user_id: Optional[str],
    json_access: JsonAccess,
) -> Optional[Dict[str, Any]]:
    """Async twin of :func:`read_session_previews` for an ``AsyncSession``."""
    row = (await sess.execute(session_stmt(sessions_table, session_id, user_id))).fetchone()
    if row is None:
        return None
    session, legacy_runs = session_from_row(row)
    resolved = resolve_preview_session_type(session, session_type)
    runs: List[Dict[str, Any]] = []
    fallback_rows: List[Any] = []
    if runs_table is not None:
        runs, missing = runs_from_rows(
            (await sess.execute(runs_stmt(runs_table, session_id, json_access))).fetchall(), resolved
        )
        if missing:
            fallback_rows = list((await sess.execute(fallback_runs_stmt(runs_table, missing, json_access))).fetchall())
    return complete_session(session, legacy_runs, runs, fallback_rows, resolved)
