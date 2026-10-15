"""Utility functions for the MongoDB database class."""

import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from agno.db.mongo.schemas import get_collection_indexes
from agno.utils.log import log_error, log_info, log_warning

try:
    from pymongo import ReturnDocument
    from pymongo.collection import Collection
    from pymongo.errors import OperationFailure
except ImportError:
    raise ImportError("`pymongo` not installed. Please install it using `pip install pymongo`")

if TYPE_CHECKING:
    from agno.db.mongo.async_mongo import AsyncMongoCollectionType


LEGACY_METRICS_INDEX = "date_1_aggregation_period_1"
INDEX_NOT_FOUND = 27  # MongoDB's IndexNotFound error code


# -- DB util methods --
def create_collection_indexes(collection: Collection, collection_type: str) -> None:
    """Create all required indexes for a collection.

    Each index is attempted independently: one conflicting index (e.g. a legacy
    collection whose index exists with different options) must not prevent the
    remaining indexes from being created.
    """
    try:
        # Before the creation loop, so an index that fails to build does not leave the
        # obsolete key in place as well
        if collection_type == "metrics" and LEGACY_METRICS_INDEX in collection.index_information():
            try:
                collection.drop_index(LEGACY_METRICS_INDEX)
                log_info(
                    f"Dropped obsolete metrics index {LEGACY_METRICS_INDEX}, superseded by the per-user unique key"
                )
            except OperationFailure as e:
                if e.code != INDEX_NOT_FOUND:
                    log_warning(f"Could not drop obsolete metrics index {LEGACY_METRICS_INDEX}: {str(e)}")

        indexes = get_collection_indexes(collection_type)
    except Exception as e:
        log_warning(f"Error creating indexes for {collection_type} collection: {str(e)}")
        return
    for index_spec in indexes:
        key = index_spec["key"]
        unique = index_spec.get("unique", False)
        extra = {"name": index_spec["name"]} if "name" in index_spec else {}

        try:
            if isinstance(key, list):
                collection.create_index(key, unique=unique, **extra)
            else:
                collection.create_index([(key, 1)], unique=unique, **extra)
        except OperationFailure as e:
            # e.g. IndexOptionsConflict on a legacy collection, or a DuplicateKey when a
            # unique index cannot be built over pre-existing duplicates: skip it, try the rest
            log_warning(f"Error creating index {key!r} for {collection_type} collection: {str(e)}")
        except Exception as e:
            # Connection-level failure: the remaining creates would all fail too
            # (each waiting out the server-selection timeout), so stop here
            log_warning(f"Error creating indexes for {collection_type} collection: {str(e)}")
            return


async def create_collection_indexes_async(collection: Any, collection_type: str) -> None:
    """Create all required indexes for a collection (async version for Motor).

    Each index is attempted independently: one conflicting index (e.g. a legacy
    collection whose index exists with different options) must not prevent the
    remaining indexes from being created.
    """
    try:
        # See create_collection_indexes: the drop goes before the creation loop
        if collection_type == "metrics" and LEGACY_METRICS_INDEX in await collection.index_information():
            try:
                await collection.drop_index(LEGACY_METRICS_INDEX)
                log_info(
                    f"Dropped obsolete metrics index {LEGACY_METRICS_INDEX}, superseded by the per-user unique key"
                )
            except OperationFailure as e:
                if e.code != INDEX_NOT_FOUND:
                    log_warning(f"Could not drop obsolete metrics index {LEGACY_METRICS_INDEX}: {str(e)}")

        indexes = get_collection_indexes(collection_type)
    except Exception as e:
        log_warning(f"Error creating indexes for {collection_type} collection: {str(e)}")
        return
    for index_spec in indexes:
        key = index_spec["key"]
        unique = index_spec.get("unique", False)
        extra = {"name": index_spec["name"]} if "name" in index_spec else {}

        try:
            if isinstance(key, list):
                await collection.create_index(key, unique=unique, **extra)
            else:
                await collection.create_index([(key, 1)], unique=unique, **extra)
        except OperationFailure as e:
            # e.g. IndexOptionsConflict on a legacy collection, or a DuplicateKey when a
            # unique index cannot be built over pre-existing duplicates: skip it, try the rest
            log_warning(f"Error creating index {key!r} for {collection_type} collection: {str(e)}")
        except Exception as e:
            # Connection-level failure: the remaining creates would all fail too
            # (each waiting out the server-selection timeout), so stop here
            log_warning(f"Error creating indexes for {collection_type} collection: {str(e)}")
            return


def apply_sorting(
    query_args: Dict[str, Any], sort_by: Optional[str] = None, sort_order: Optional[str] = None
) -> List[tuple]:
    """Apply sorting to MongoDB query."""
    if sort_by is None:
        return []

    sort_direction = 1 if sort_order == "asc" else -1
    return [(sort_by, sort_direction)]


def apply_pagination(
    query_args: Dict[str, Any], limit: Optional[int] = None, page: Optional[int] = None
) -> Dict[str, Any]:
    """Apply pagination to MongoDB query."""
    if limit is not None:
        query_args["limit"] = limit
        if page is not None:
            query_args["skip"] = (page - 1) * limit
    return query_args


# -- Metrics util methods --
def calculate_date_metrics(date_to_process: date, sessions_data: dict) -> List[dict]:
    """Calculate metrics for the given single date, one record per user.

    Sessions with no ``user_id`` are bucketed under the empty-string sentinel.
    """

    def _empty_metric_record() -> Dict[str, Any]:
        return {
            "users_count": 0,
            "agent_sessions_count": 0,
            "team_sessions_count": 0,
            "workflow_sessions_count": 0,
            "agent_runs_count": 0,
            "team_runs_count": 0,
            "workflow_runs_count": 0,
            "token_metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "audio_total_tokens": 0,
                "audio_input_tokens": 0,
                "audio_output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            },
            "model_counts": {},
        }

    session_types = [
        ("agent", "agent_sessions_count", "agent_runs_count"),
        ("team", "team_sessions_count", "team_runs_count"),
        ("workflow", "workflow_sessions_count", "workflow_runs_count"),
    ]

    per_user: Dict[str, Dict[str, Any]] = {}

    for session_type, sessions_count_key, runs_count_key in session_types:
        sessions = sessions_data.get(session_type, []) or []

        for session in sessions:
            bucket_key = session.get("user_id") or ""
            bucket = per_user.setdefault(bucket_key, _empty_metric_record())
            bucket[sessions_count_key] += 1

            runs = session.get("runs", []) or []
            if isinstance(runs, str):
                runs = json.loads(runs)
            bucket[runs_count_key] += len(runs)
            for run in runs:
                if model_id := run.get("model"):
                    model_provider = run.get("model_provider", "")
                    key = f"{model_id}:{model_provider}"
                    bucket["model_counts"][key] = bucket["model_counts"].get(key, 0) + 1

            session_data = session.get("session_data", {}) or {}
            if isinstance(session_data, str):
                session_data = json.loads(session_data)
            session_metrics = session_data.get("session_metrics", {}) or {}
            for field in bucket["token_metrics"]:
                bucket["token_metrics"][field] += session_metrics.get(field, 0)

    current_time = int(time.time())
    completed = date_to_process < datetime.now(timezone.utc).date()

    records: List[dict] = []
    for user_id, bucket in per_user.items():
        model_metrics = []
        for model, count in bucket["model_counts"].items():
            model_id, model_provider = model.rsplit(":", 1)
            model_metrics.append({"model_id": model_id, "model_provider": model_provider, "count": count})

        users_count = 0 if user_id == "" else 1

        records.append(
            {
                "id": str(uuid4()),
                "date": date_to_process,
                "completed": completed,
                "token_metrics": bucket["token_metrics"],
                "model_metrics": model_metrics,
                "created_at": current_time,
                "updated_at": current_time,
                "aggregation_period": "daily",
                "user_id": user_id,
                "users_count": users_count,
                "agent_sessions_count": bucket["agent_sessions_count"],
                "team_sessions_count": bucket["team_sessions_count"],
                "workflow_sessions_count": bucket["workflow_sessions_count"],
                "agent_runs_count": bucket["agent_runs_count"],
                "team_runs_count": bucket["team_runs_count"],
                "workflow_runs_count": bucket["workflow_runs_count"],
            }
        )

    return records


def fetch_all_sessions_data(
    sessions: List[Dict[str, Any]], dates_to_process: list[date], start_timestamp: int
) -> Optional[dict]:
    """Return all session data for the given dates, for all session types."""
    if not dates_to_process:
        return None

    all_sessions_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        date_to_process.isoformat(): {"agent": [], "team": [], "workflow": []} for date_to_process in dates_to_process
    }

    for session in sessions:
        session_date = (
            datetime.fromtimestamp(session.get("created_at", start_timestamp), tz=timezone.utc).date().isoformat()
        )
        if session_date in all_sessions_data:
            all_sessions_data[session_date][session["session_type"]].append(session)

    return all_sessions_data


def get_dates_to_calculate_metrics_for(starting_date: date) -> list[date]:
    """Return the list of dates to calculate metrics for."""
    today = datetime.now(timezone.utc).date()
    days_diff = (today - starting_date).days + 1
    if days_diff <= 0:
        return []
    return [starting_date + timedelta(days=x) for x in range(days_diff)]


def bulk_upsert_metrics(collection: Collection, metrics_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bulk upsert metrics into the database.

    Args:
        collection (Collection): The collection to upsert the metrics into.
        metrics_records (List[Dict[str, Any]]): The list of metrics records to upsert.

    Returns:
        The list of upserted metrics records.
    """
    if not metrics_records:
        return []

    results = []
    for record in metrics_records:
        record["date"] = record["date"].isoformat() if isinstance(record["date"], date) else record["date"]
        try:
            # Legacy records have no ``user_id``, so default to the empty-string sentinel bucket
            key_filter = {
                "user_id": record.get("user_id", ""),
                "date": record["date"],
                "aggregation_period": record["aggregation_period"],
            }

            # id and created_at belong to the first write, so only $set the fields that change
            identity = {"id": record.pop("id"), "created_at": record.pop("created_at")}
            stored = collection.find_one_and_update(
                key_filter,
                {"$set": record, "$setOnInsert": identity},
                upsert=True,
                return_document=ReturnDocument.AFTER,
                projection={"_id": False},
            )
            # Upserting and returning AFTER always yields the document; a miss would put a
            # None where every caller expects a record
            if stored is not None:
                results.append(stored)

        except Exception as e:
            log_error(f"Error upserting metrics record: {str(e)}")
            continue

    return results


async def abulk_upsert_metrics(
    collection: "AsyncMongoCollectionType", metrics_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Async bulk upsert metrics into the database.

    Args:
        collection (AsyncMongoCollectionType): The async collection to upsert the metrics into.
        metrics_records (List[Dict[str, Any]]): The list of metrics records to upsert.

    Returns:
        The list of upserted metrics records.
    """
    if not metrics_records:
        return []

    results = []
    for record in metrics_records:
        record["date"] = record["date"].isoformat() if isinstance(record["date"], date) else record["date"]
        try:
            # Legacy records have no ``user_id``, so default to the empty-string sentinel bucket
            key_filter = {
                "user_id": record.get("user_id", ""),
                "date": record["date"],
                "aggregation_period": record["aggregation_period"],
            }

            # id and created_at belong to the first write, so only $set the fields that change
            identity = {"id": record.pop("id"), "created_at": record.pop("created_at")}
            stored = await collection.find_one_and_update(
                key_filter,
                {"$set": record, "$setOnInsert": identity},
                upsert=True,
                return_document=ReturnDocument.AFTER,
                projection={"_id": False},
            )
            # Upserting and returning AFTER always yields the document; a miss would put a
            # None where every caller expects a record
            if stored is not None:
                results.append(stored)

        except Exception as e:
            log_error(f"Error upserting metrics record: {str(e)}")
            continue

    return results


# -- Run previews --
def run_preview_pipeline(session_id: str) -> List[Dict[str, Any]]:
    """Aggregation reading only the preview keys out of each run's ``run_data``, in insertion order."""
    from agno.db.utils import RUN_PREVIEW_KEYS

    project: Dict[str, Any] = {"_id": 0, **{key: f"$run_data.{key}" for key in RUN_PREVIEW_KEYS}}
    project["input_content"] = "$run_data.input.input_content"
    # A workflow run previews its raw input, which is small; other runs carry media beside input_content.
    project["workflow_input"] = {
        "$cond": [
            {"$in": [{"$type": "$run_data.workflow_id"}, ["missing", "null"]]},
            "$$REMOVE",
            "$run_data.input",
        ]
    }
    return [
        {"$match": {"session_id": session_id}},
        {"$addFields": {"_ri": {"$ifNull": ["$run_index", 0]}, "_ca": {"$ifNull": ["$created_at", 0]}}},
        {"$sort": {"_ri": 1, "_ca": 1}},
        {"$project": project},
    ]


def run_preview_fallback_query(run_ids: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter and projection for what the input preview falls back to, for runs that stored no usable input."""
    from agno.db.utils import PREVIEW_FALLBACK_KEYS

    projection: Dict[str, Any] = {"_id": 0, "run_id": 1, **{f"run_data.{key}": 1 for key in PREVIEW_FALLBACK_KEYS}}
    return {"run_id": {"$in": run_ids}}, projection


def session_preview_query(session_id: str, user_id: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter and projection for a session's scoping/type fields plus any legacy ``runs`` blob."""
    from agno.db.utils import SESSION_PREVIEW_KEYS

    query: Dict[str, Any] = {"session_id": session_id}
    if user_id is not None:
        query["user_id"] = user_id
    return query, {"_id": 0, "runs": 1, **{key: 1 for key in SESSION_PREVIEW_KEYS}}


def fallbacks_from_docs(docs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        doc["run_id"]: {key: value for key, value in (doc.get("run_data") or {}).items() if value is not None}
        for doc in docs
    }


def preview_session_from_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """Session preview dict and the raw legacy ``runs`` blob from a ``session_preview_query`` doc."""
    from agno.db.utils import SESSION_PREVIEW_KEYS

    return {key: doc.get(key) for key in SESSION_PREVIEW_KEYS}, doc.get("runs")
