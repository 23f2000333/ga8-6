import copy
import hashlib
import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ------------------------------------------------------------
# Global process state
#
# sessions[session_id] contains ALL state for that session.
# Nothing from one session is ever read by another session.
# ------------------------------------------------------------

sessions: Dict[str, Dict[str, Any]] = {}

NODES = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

NODE_INDEX = {node: i for i, node in enumerate(NODES)}

INPUT_NAMES = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]

STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def compact_json(value: Any) -> str:
    """
    Compact canonical JSON used for conflict comparisons
    and event identity.

    sort_keys=True makes object key ordering deterministic.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json_array(values):
    payload = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def sha256_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_positive_integer(value) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and value <= 9007199254740991
    )


def new_node_state():
    return {
        "status": None,
        "attempt": None,
        "artifactDigest": None,
        "startEventId": None,
        "successEventId": None,
        "terminalEventId": None,
        "retryEventId": None,
    }


def new_session(inputs, revision, input_signature):
    return {
        "revision": revision,

        # Current revision's inputs
        "inputs": copy.deepcopy(inputs),
        "input_signature": input_signature,

        # Current execution state
        "nodes": {
            node: new_node_state()
            for node in NODES
        },

        # Successful content-addressed cache.
        #
        # key -> {
        #   artifactDigest,
        #   eventId
        # }
        #
        # This survives revision changes within this session.
        "cache": {},

        # eventId -> canonical compact event JSON
        "events": {},

        # First successful artifact permanently bound to a key.
        #
        # key -> {
        #   artifactDigest,
        #   eventId
        # }
        "evidence": {},
    }


def get_parent(node):
    idx = NODE_INDEX[node]
    if idx == 0:
        return None
    return NODES[idx - 1]


def event_key_is_valid(node, event):
    return isinstance(event.get("key"), str) and bool(event["key"])


def artifact_is_valid(event):
    status = event["status"]

    if status == "succeeded":
        return (
            isinstance(event["artifactDigest"], str)
            and bool(event["artifactDigest"])
        )

    return event["artifactDigest"] is None


def receipt_is_valid(event):
    node = event["node"]
    status = event["status"]

    if node in ("register", "publish") and status == "succeeded":
        expected = f"receipt:{node}:{event['key']}"
        return event["receiptId"] == expected

    return event["receiptId"] is None


def event_shape_valid(event):
    """
    Events must contain exactly the eight listed fields.
    """
    expected = {
        "eventId",
        "revision",
        "node",
        "attempt",
        "status",
        "key",
        "artifactDigest",
        "receiptId",
    }

    if not isinstance(event, dict):
        return False

    if set(event.keys()) != expected:
        return False

    if not isinstance(event["eventId"], str) or not event["eventId"]:
        return False

    if not safe_positive_integer(event["revision"]):
        return False

    if event["node"] not in NODE_INDEX:
        return False

    if not safe_positive_integer(event["attempt"]):
        return False

    if event["status"] not in STATUSES:
        return False

    if not event_key_is_valid(event["node"], event):
        return False

    if not artifact_is_valid(event):
        return False

    if not receipt_is_valid(event):
        return False

    return True


# ------------------------------------------------------------
# Cache key construction
# ------------------------------------------------------------

def get_dependency_values(session, node):
    inputs = session["inputs"]
    states = session["nodes"]

    if node == "verify_data":
        return [
            inputs["generation"],
            inputs["checksum"],
        ]

    if node == "prepare":
        return [
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ]

    if node == "train":
        return [
            states["prepare"]["artifactDigest"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    if node == "evaluate":
        return [
            states["train"]["artifactDigest"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    if node == "register":
        return [
            states["evaluate"]["artifactDigest"],
            inputs["schemaDigest"],
        ]

    if node == "publish":
        return [
            states["register"]["artifactDigest"],
            inputs["publishConfig"],
        ]

    raise ValueError(node)


def compute_key(session, node):
    deps = get_dependency_values(session, node)

    # Downstream key is null until parent artifact exists.
    if any(value is None for value in deps):
        return None

    return sha256_json_array(deps)


# ------------------------------------------------------------
# Dependency digest response
# ------------------------------------------------------------

def dependency_digest_map(session, node, cache_key):
    inputs = session["inputs"]
    states = session["nodes"]

    result = {}

    if node == "verify_data":
        result["generation"] = sha256_string(inputs["generation"])
        result["checksum"] = sha256_string(inputs["checksum"])

    elif node == "prepare":
        result["canonicalData"] = sha256_string(
            inputs["canonicalData"]
        )
        result["prepareCode"] = sha256_string(
            inputs["prepareCode"]
        )
        result["prepareConfig"] = sha256_string(
            inputs["prepareConfig"]
        )

    elif node == "train":
        result["prepareArtifact"] = states["prepare"]["artifactDigest"]
        result["trainCode"] = sha256_string(
            inputs["trainCode"]
        )
        result["trainConfig"] = sha256_string(
            inputs["trainConfig"]
        )
        result["runtime"] = sha256_string(
            inputs["runtime"]
        )

    elif node == "evaluate":
        result["trainArtifact"] = states["train"]["artifactDigest"]
        result["canonicalData"] = sha256_string(
            inputs["canonicalData"]
        )
        result["evaluateCode"] = sha256_string(
            inputs["evaluateCode"]
        )
        result["evaluateConfig"] = sha256_string(
            inputs["evaluateConfig"]
        )

    elif node == "register":
        result["evaluateArtifact"] = states["evaluate"]["artifactDigest"]
        result["schemaDigest"] = sha256_string(
            inputs["schemaDigest"]
        )

    elif node == "publish":
        result["registerArtifact"] = states["register"]["artifactDigest"]
        result["publishConfig"] = sha256_string(
            inputs["publishConfig"]
        )

    result["cacheKey"] = cache_key

    return result


# ------------------------------------------------------------
# Parent readiness
# ------------------------------------------------------------

def parent_status(session, node):
    parent = get_parent(node)

    if parent is None:
        return "ready"

    state = session["nodes"][parent]

    if state["status"] == "terminal_failed":
        return "terminal"

    if state["status"] != "succeeded":
        return "pending"

    return "ready"


# ------------------------------------------------------------
# Process one event
#
# Returns:
#   "accepted"
#   "ignored"
#   "status_conflict"
#   "evidence_conflict"
# ------------------------------------------------------------

def process_event(session, event):
    node = event["node"]
    state = session["nodes"][node]

    # Wrong revision is ignored.
    if event["revision"] != session["revision"]:
        return "ignored"

    current_key = compute_key(session, node)

    # Parent unavailable or wrong key => ignored.
    if current_key is None:
        return "ignored"

    if event["key"] != current_key:
        return "ignored"

    # --------------------------------------------------------
    # Check immutable successful evidence first.
    # --------------------------------------------------------

    evidence = session["evidence"].get(current_key)

    if evidence is not None:
        if event["status"] == "succeeded":
            if event["artifactDigest"] != evidence["artifactDigest"]:
                return "evidence_conflict"

            # Exact same evidence event can be replayed.
            if event["eventId"] == evidence["eventId"]:
                return "ignored"

        return "status_conflict"

    # --------------------------------------------------------
    # Current state machine
    # --------------------------------------------------------

    status = state["status"]
    attempt = event["attempt"]

    # No state yet.
    if status is None:
        if event["status"] == "started" and attempt == 1:
            state["status"] = "started"
            state["attempt"] = 1
            state["startEventId"] = event["eventId"]
            return "accepted"

        # Completion or attempt > 1 before first start is ignored.
        return "ignored"

    # started(n)
    if status == "started":
        if (
            event["attempt"] == attempt
            and event["status"] in {
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            }
        ):
            if event["status"] == "succeeded":
                state["status"] = "succeeded"
                state["artifactDigest"] = event["artifactDigest"]
                state["successEventId"] = event["eventId"]

                # Immutable evidence binding.
                session["evidence"][current_key] = {
                    "artifactDigest": event["artifactDigest"],
                    "eventId": event["eventId"],
                }

                session["cache"][current_key] = {
                    "artifactDigest": event["artifactDigest"],
                    "eventId": event["eventId"],
                }

            elif event["status"] == "retryable_failed":
                state["status"] = "retryable_failed"
                state["retryEventId"] = event["eventId"]

            else:
                state["status"] = "terminal_failed"
                state["terminalEventId"] = event["eventId"]

            return "accepted"

        # Lower attempts are ignored.
        if event["attempt"] < attempt:
            return "ignored"

        return "status_conflict"

    # retryable_failed(n)
    if status == "retryable_failed":
        if (
            event["status"] == "started"
            and event["attempt"] == attempt + 1
        ):
            state["status"] = "started"
            state["attempt"] = event["attempt"]
            state["startEventId"] = event["eventId"]
            return "accepted"

        if event["attempt"] < attempt:
            return "ignored"

        return "status_conflict"

    # terminal_failed
    if status == "terminal_failed":
        return "status_conflict"

    # succeeded
    if status == "succeeded":
        if (
            event["status"] == "succeeded"
            and event["artifactDigest"]
            != state["artifactDigest"]
        ):
            return "evidence_conflict"

        return "status_conflict"

    return "status_conflict"


# ------------------------------------------------------------
# Response node calculation
# ------------------------------------------------------------

def build_node_response(session, node):
    state = session["nodes"][node]
    cache_key = compute_key(session, node)

    dependency_digests = dependency_digest_map(
        session,
        node,
        cache_key,
    ) if cache_key is not None else {}

    triggering = []

    # --------------------------------------------------------
    # Cached success
    # --------------------------------------------------------

    if cache_key is not None and cache_key in session["cache"]:
        cached = session["cache"][cache_key]

        triggering = [cached["eventId"]]

        return {
            "node": node,
            "action": "reuse",
            "reasonCodes": ["CACHE_HIT"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": triggering,
        }

    # --------------------------------------------------------
    # Terminal upstream
    # --------------------------------------------------------

    pstatus = parent_status(session, node)

    if pstatus == "terminal":
        parent = get_parent(node)

        parent_state = session["nodes"][parent]

        triggering = (
            [parent_state["terminalEventId"]]
            if parent_state["terminalEventId"]
            else []
        )

        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["UPSTREAM_TERMINAL"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": triggering,
        }

    # --------------------------------------------------------
    # Pending upstream
    # --------------------------------------------------------

    if pstatus == "pending":
        parent = get_parent(node)
        parent_state = session["nodes"][parent]

        if parent_state["startEventId"]:
            triggering = [parent_state["startEventId"]]

        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["UPSTREAM_PENDING"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": triggering,
        }

    # --------------------------------------------------------
    # Current node state
    # --------------------------------------------------------

    if state["status"] == "terminal_failed":
        triggering = (
            [state["terminalEventId"]]
            if state["terminalEventId"]
            else []
        )

        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["TERMINAL_FAILURE"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": triggering,
        }

    if state["status"] == "started":
        triggering = (
            [state["startEventId"]]
            if state["startEventId"]
            else []
        )

        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["RUNNING"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": triggering,
        }

    if state["status"] == "retryable_failed":
        triggering = (
            [state["retryEventId"]]
            if state["retryEventId"]
            else []
        )

        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": ["RETRYABLE_FAILURE"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": triggering,
        }

    # --------------------------------------------------------
    # Ready without cache
    # --------------------------------------------------------

    if cache_key is not None:
        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": ["CACHE_MISS"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": [],
        }

    # Root should normally never reach here.
    return {
        "node": node,
        "action": "block",
        "reasonCodes": ["UPSTREAM_PENDING"],
        "dependencyDigests": dependency_digests,
        "triggeringEventIds": [],
    }


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

async def parse_request(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise ValueError("INVALID_REQUEST")

    if not isinstance(body, dict):
        raise ValueError("INVALID_REQUEST")

    required = {
        "session",
        "revision",
        "inputs",
        "events",
    }

    if not required.issubset(body.keys()):
        raise ValueError("INVALID_REQUEST")

    session_id = body["session"]

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("INVALID_REQUEST")

    if not safe_positive_integer(body["revision"]):
        raise ValueError("INVALID_REQUEST")

    if not isinstance(body["inputs"], dict):
        raise ValueError("INVALID_REQUEST")

    if not isinstance(body["events"], list):
        raise ValueError("INVALID_REQUEST")

    # All required inputs must exist and be non-empty strings.
    for name in INPUT_NAMES:
        value = body["inputs"].get(name)

        if not isinstance(value, str) or not value:
            raise ValueError("INVALID_REQUEST")

    return body


# ------------------------------------------------------------
# POST /pipeline
# ------------------------------------------------------------

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await parse_request(request)
    except ValueError as exc:
        return JSONResponse(
            status_code=409,
            content={"error": str(exc)},
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # --------------------------------------------------------
    # First request for this session
    # --------------------------------------------------------

    if session_id not in sessions:
        session = new_session(
            inputs,
            revision,
            compact_json(inputs),
        )
        sessions[session_id] = session

    else:
        session = sessions[session_id]

        # ----------------------------------------------------
        # Same revision
        # ----------------------------------------------------

        if revision == session["revision"]:

            if compact_json(inputs) != session["input_signature"]:
                return JSONResponse(
                    status_code=409,
                    content={"error": "REVISION_CONFLICT"},
                )

        # ----------------------------------------------------
        # New revision
        # ----------------------------------------------------

        elif revision > session["revision"]:

            session["revision"] = revision
            session["inputs"] = copy.deepcopy(inputs)
            session["input_signature"] = compact_json(inputs)

            # Clear attempt / terminal / current execution state.
            session["nodes"] = {
                node: new_node_state()
                for node in NODES
            }

            # IMPORTANT:
            # cache and immutable evidence survive.
            #
            # event IDs remain global within this session so
            # old IDs cannot be reused with different content.
            #
            # events themselves remain in session["events"].

        else:
            # Request itself can still be valid; events from an
            # older revision are ignored. However the request's
            # revision cannot become the current revision.
            #
            # The specification says old-revision events are
            # ignored, so we build a temporary interpretation
            # only when needed.
            pass

    # --------------------------------------------------------
    # Atomic batch:
    #
    # Work on a deep copy. If ANY event produces a 409,
    # discard the entire copy.
    # --------------------------------------------------------

    working = copy.deepcopy(session)

    accepted = []
    ignored = []

    for event in events:

        # Structural validation.
        if not event_shape_valid(event):
            return JSONResponse(
                status_code=409,
                content={"error": "INVALID_EVENT"},
            )

        event_id = event["eventId"]
        canonical_event = compact_json(event)

        # ----------------------------------------------------
        # Global event ID handling.
        # ----------------------------------------------------

        if event_id in working["events"]:

            if working["events"][event_id] == canonical_event:
                # Exact replay ignored.
                ignored.append(event_id)
                continue

            return JSONResponse(
                status_code=409,
                content={"error": "EVENT_ID_CONFLICT"},
            )

        # ----------------------------------------------------
        # Wrong revision is ignored and does not consume ID.
        # ----------------------------------------------------

        if event["revision"] != working["revision"]:
            ignored.append(event_id)
            continue

        # ----------------------------------------------------
        # Process transition.
        # ----------------------------------------------------

        result = process_event(working, event)

        if result == "ignored":
            ignored.append(event_id)

            # Ignored events DO NOT consume their IDs.
            continue

        if result == "status_conflict":
            return JSONResponse(
                status_code=409,
                content={"error": "STATUS_CONFLICT"},
            )

        if result == "evidence_conflict":
            return JSONResponse(
                status_code=409,
                content={"error": "EVIDENCE_CONFLICT"},
            )

        # Accepted event consumes its ID.
        working["events"][event_id] = canonical_event
        accepted.append(event_id)

    # --------------------------------------------------------
    # Entire batch commits atomically here.
    # --------------------------------------------------------

    sessions[session_id] = working

    # --------------------------------------------------------
    # Build response in DAG order.
    # --------------------------------------------------------

    node_responses = [
        build_node_response(working, node)
        for node in NODES
    ]

    return {
        "revision": working["revision"],
        "acceptedEventIds": accepted,
        "ignoredEventIds": ignored,
        "nodes": node_responses,
    }


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "pipeline-controller",
    }
