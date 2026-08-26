import copy
import hashlib
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


# ============================================================
# CONSTANTS
# ============================================================

NODES = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

PARENT = {
    "verify_data": None,
    "prepare": "verify_data",
    "train": "prepare",
    "evaluate": "train",
    "register": "evaluate",
    "publish": "register",
}

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

MAX_SAFE_INTEGER = 9007199254740991


# ============================================================
# GLOBAL PROCESS STATE
#
# Session isolation:
#
# sessions[session_id] = completely independent state
#
# Nothing in cache/evidence/nodes is shared between sessions.
# ============================================================

sessions = {}


# ============================================================
# JSON / HASH HELPERS
# ============================================================

def compact_json(value):
    """
    Canonical compact JSON representation.

    sort_keys=True is used for objects so that equivalent object
    representations have the same canonical form.

    Arrays retain their exact supplied order.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_compact_array(values):
    """
    SHA-256 over UTF-8 compact JSON array.

    IMPORTANT:
    The array order is NEVER changed.
    """
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def sha256_string(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def positive_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and value <= MAX_SAFE_INTEGER
    )


# ============================================================
# NODE STATE
# ============================================================

def empty_node_state():
    return {
        "status": None,
        "attempt": None,
        "artifactDigest": None,

        "startEventId": None,
        "retryEventId": None,
        "successEventId": None,
        "terminalEventId": None,
    }


# ============================================================
# SESSION STATE
# ============================================================

def new_session(inputs, revision):
    return {
        "revision": revision,

        # COMPLETE input object, including extra metadata.
        "inputs": copy.deepcopy(inputs),

        # Used for same-revision conflict detection.
        "inputSignature": compact_json(inputs),

        # Execution state for CURRENT revision only.
        "nodes": {
            node: empty_node_state()
            for node in NODES
        },

        # Successful content-addressed cache.
        #
        # (node, key) -> {
        #     artifactDigest,
        #     eventId
        # }
        #
        # This survives revisions.
        "cache": {},

        # First-success immutable evidence.
        #
        # (node, key) -> {
        #     artifactDigest,
        #     eventId
        # }
        #
        # This survives revisions.
        "evidence": {},

        # Global event IDs within this session.
        #
        # eventId -> canonical compact event JSON
        "events": {},
    }


def cache_id(node, key):
    return (node, key)


# ============================================================
# REUSABLE ARTIFACT
# ============================================================

def reusable_artifact(session, node):
    """
    A node's artifact is reusable when:

    1. It succeeded in the current revision, OR
    2. Its exact content-addressed cache entry exists.

    For downstream nodes, its cache key is computed using the
    parent's reusable artifact.
    """

    state = session["nodes"][node]

    # Current revision success.
    if state["status"] == "succeeded":
        return state["artifactDigest"]

    # Otherwise see if this node's content-addressed result
    # exists in the persistent session cache.
    key = compute_key(session, node)

    if key is None:
        return None

    cached = session["cache"].get(
        cache_id(node, key)
    )

    if cached is None:
        return None

    return cached["artifactDigest"]


# ============================================================
# EXACT DEPENDENCY ARRAYS
# ============================================================

def dependency_array(session, node):
    """
    EXACT arrays from the specification.

    verify_data:
        [generation, checksum]

    prepare:
        [canonicalData, prepareCode, prepareConfig]

    train:
        [prepareArtifact, trainCode, trainConfig, runtime]

    evaluate:
        [trainArtifact, canonicalData,
         evaluateCode, evaluateConfig]

    register:
        [evaluateArtifact, schemaDigest]

    publish:
        [registerArtifact, publishConfig]
    """

    inputs = session["inputs"]

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
        parent_artifact = reusable_artifact(
            session,
            "prepare",
        )

        if parent_artifact is None:
            return None

        return [
            parent_artifact,
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    if node == "evaluate":
        parent_artifact = reusable_artifact(
            session,
            "train",
        )

        if parent_artifact is None:
            return None

        return [
            parent_artifact,
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    if node == "register":
        parent_artifact = reusable_artifact(
            session,
            "evaluate",
        )

        if parent_artifact is None:
            return None

        return [
            parent_artifact,
            inputs["schemaDigest"],
        ]

    if node == "publish":
        parent_artifact = reusable_artifact(
            session,
            "register",
        )

        if parent_artifact is None:
            return None

        return [
            parent_artifact,
            inputs["publishConfig"],
        ]

    return None


def compute_key(session, node):
    """
    Parent-gated content address.

    No parent artifact => downstream key is None.
    """

    deps = dependency_array(
        session,
        node,
    )

    if deps is None:
        return None

    return sha256_compact_array(deps)


def cache_hit(session, node):
    key = compute_key(session, node)

    if key is None:
        return None

    return session["cache"].get(
        cache_id(node, key)
    )


# ============================================================
# DEPENDENCY DIGEST RESPONSE
# ============================================================

def dependency_digests(session, node, key):
    inputs = session["inputs"]

    result = {}

    if node == "verify_data":

        result["generation"] = sha256_string(
            inputs["generation"]
        )

        result["checksum"] = sha256_string(
            inputs["checksum"]
        )

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

        result["prepareArtifact"] = reusable_artifact(
            session,
            "prepare",
        )

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

        result["trainArtifact"] = reusable_artifact(
            session,
            "train",
        )

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

        result["evaluateArtifact"] = reusable_artifact(
            session,
            "evaluate",
        )

        result["schemaDigest"] = sha256_string(
            inputs["schemaDigest"]
        )

    elif node == "publish":

        result["registerArtifact"] = reusable_artifact(
            session,
            "register",
        )

        result["publishConfig"] = sha256_string(
            inputs["publishConfig"]
        )

    result["cacheKey"] = key

    return result


# ============================================================
# EVENT STRUCTURE VALIDATION
# ============================================================

EVENT_FIELDS = {
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
}


def event_structure_valid(event):
    """
    Structural errors are INVALID_EVENT.

    Semantic bad values such as invalid attempt/status/artifact/
    receipt are handled separately and are ignored as required.
    """

    if not isinstance(event, dict):
        return False

    if set(event.keys()) != EVENT_FIELDS:
        return False

    if (
        not isinstance(event["eventId"], str)
        or not event["eventId"]
    ):
        return False

    if not positive_safe_integer(
        event["revision"]
    ):
        return False

    return True


def event_semantically_valid(event):
    """
    Returns True only for an event whose semantic fields are valid.

    The specification says invalid status/artifact/receipt/attempt
    are ignored rather than causing 409.
    """

    if event["node"] not in NODES:
        return False

    if not positive_safe_integer(
        event["attempt"]
    ):
        return False

    if event["status"] not in STATUSES:
        return False

    if (
        not isinstance(event["key"], str)
        or not event["key"]
    ):
        return False

    # Success requires non-empty artifact.
    if event["status"] == "succeeded":

        if (
            not isinstance(
                event["artifactDigest"],
                str,
            )
            or not event["artifactDigest"]
        ):
            return False

    # Every other status requires null artifact.
    else:

        if event["artifactDigest"] is not None:
            return False

    # Register/publish successful event must have exact receipt.
    if (
        event["node"] in {
            "register",
            "publish",
        }
        and event["status"] == "succeeded"
    ):

        expected = (
            f"receipt:{event['node']}:{event['key']}"
        )

        if event["receiptId"] != expected:
            return False

    else:

        if event["receiptId"] is not None:
            return False

    return True


# ============================================================
# EVENT STATE MACHINE
# ============================================================

def apply_event(session, event):
    """
    Returns:

        accepted
        ignored
        status_conflict
        evidence_conflict
    """

    node = event["node"]

    # Wrong node is ignored.
    if node not in NODES:
        return "ignored"

    # Wrong revision is ignored.
    if event["revision"] != session["revision"]:
        return "ignored"

    key = compute_key(
        session,
        node,
    )

    # Parent unavailable.
    if key is None:
        return "ignored"

    # Wrong key.
    if event["key"] != key:
        return "ignored"

    cid = cache_id(
        node,
        key,
    )

    state = session["nodes"][node]

    # --------------------------------------------------------
    # Existing immutable successful evidence.
    # --------------------------------------------------------

    evidence = session["evidence"].get(cid)

    if evidence is not None:

        # Different successful artifact.
        if (
            event["status"] == "succeeded"
            and event["artifactDigest"]
            != evidence["artifactDigest"]
        ):
            return "evidence_conflict"

        # Same artifact but new event:
        # successful/current-cache state rejects all new events.
        return "status_conflict"

    # --------------------------------------------------------
    # No current state.
    # --------------------------------------------------------

    if state["status"] is None:

        # Only started(1) is accepted.
        if (
            event["status"] == "started"
            and event["attempt"] == 1
        ):

            state["status"] = "started"
            state["attempt"] = 1
            state["startEventId"] = event["eventId"]

            return "accepted"

        # Completion or attempt > 1 without start:
        # explicitly ignored.
        return "ignored"

    # --------------------------------------------------------
    # started(n)
    # --------------------------------------------------------

    if state["status"] == "started":

        # Valid completion of the current attempt.
        if (
            event["attempt"] == state["attempt"]
            and event["status"] in {
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            }
        ):

            if event["status"] == "succeeded":

                state["status"] = "succeeded"

                state["attempt"] = event["attempt"]

                state["artifactDigest"] = (
                    event["artifactDigest"]
                )

                state["successEventId"] = (
                    event["eventId"]
                )

                # Immutable first successful binding.
                session["evidence"][cid] = {
                    "artifactDigest": (
                        event["artifactDigest"]
                    ),
                    "eventId": event["eventId"],
                }

                # Successful cache survives revisions.
                session["cache"][cid] = {
                    "artifactDigest": (
                        event["artifactDigest"]
                    ),
                    "eventId": event["eventId"],
                }

                return "accepted"

            if event["status"] == "retryable_failed":

                state["status"] = (
                    "retryable_failed"
                )

                state["retryEventId"] = (
                    event["eventId"]
                )

                return "accepted"

            state["status"] = "terminal_failed"

            state["terminalEventId"] = (
                event["eventId"]
            )

            return "accepted"

        # Lower attempt is ignored.
        if event["attempt"] < state["attempt"]:
            return "ignored"

        # Any other transition conflicts.
        return "status_conflict"

    # --------------------------------------------------------
    # retryable_failed(n)
    # --------------------------------------------------------

    if state["status"] == "retryable_failed":

        # Exactly n+1 started is accepted.
        if (
            event["status"] == "started"
            and event["attempt"]
            == state["attempt"] + 1
        ):

            state["status"] = "started"

            state["attempt"] = event["attempt"]

            state["startEventId"] = (
                event["eventId"]
            )

            return "accepted"

        # Lower attempt ignored.
        if event["attempt"] < state["attempt"]:
            return "ignored"

        return "status_conflict"

    # --------------------------------------------------------
    # terminal_failed
    # --------------------------------------------------------

    if state["status"] == "terminal_failed":
        return "status_conflict"

    # --------------------------------------------------------
    # succeeded
    # --------------------------------------------------------

    if state["status"] == "succeeded":

        if (
            event["status"] == "succeeded"
            and event["artifactDigest"]
            != state["artifactDigest"]
        ):
            return "evidence_conflict"

        return "status_conflict"

    return "status_conflict"


# ============================================================
# RESPONSE NODE
# ============================================================

def make_node_response(session, node):
    state = session["nodes"][node]

    key = compute_key(
        session,
        node,
    )

    deps = (
        dependency_digests(
            session,
            node,
            key,
        )
        if key is not None
        else {}
    )

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    cached = cache_hit(
        session,
        node,
    )

    if cached is not None:

        return {
            "node": node,
            "action": "reuse",
            "reasonCodes": [
                "CACHE_HIT"
            ],
            "dependencyDigests": deps,
            "triggeringEventIds": [
                cached["eventId"]
            ],
        }

    # --------------------------------------------------------
    # CURRENT NODE TERMINAL
    # --------------------------------------------------------

    if state["status"] == "terminal_failed":

        return {
            "node": node,
            "action": "block",
            "reasonCodes": [
                "TERMINAL_FAILURE"
            ],
            "dependencyDigests": deps,
            "triggeringEventIds": [
                state["terminalEventId"]
            ],
        }

    # --------------------------------------------------------
    # CURRENT NODE RUNNING
    # --------------------------------------------------------

    if state["status"] == "started":

        return {
            "node": node,
            "action": "block",
            "reasonCodes": [
                "RUNNING"
            ],
            "dependencyDigests": deps,
            "triggeringEventIds": [
                state["startEventId"]
            ],
        }

    # --------------------------------------------------------
    # CURRENT NODE RETRYABLE FAILURE
    # --------------------------------------------------------

    if state["status"] == "retryable_failed":

        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": [
                "RETRYABLE_FAILURE"
            ],
            "dependencyDigests": deps,
            "triggeringEventIds": [
                state["retryEventId"]
            ],
        }

    # --------------------------------------------------------
    # PARENT GATING
    # --------------------------------------------------------

    parent = PARENT[node]

    if parent is not None:

        parent_state = session["nodes"][parent]

        # A parent terminal failure propagates downstream.
        if parent_state["status"] == "terminal_failed":

            return {
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_TERMINAL"
                ],
                "dependencyDigests": deps,
                "triggeringEventIds": [
                    parent_state["terminalEventId"]
                ],
            }

        # Parent must be reusable before this node can have a key.
        if reusable_artifact(
            session,
            parent,
        ) is None:

            triggering = []

            # A pending/running parent can expose its start event.
            if parent_state["startEventId"] is not None:
                triggering = [
                    parent_state["startEventId"]
                ]

            return {
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests": deps,
                "triggeringEventIds": triggering,
            }

    # --------------------------------------------------------
    # READY WITHOUT CACHE
    # --------------------------------------------------------

    if key is not None:

        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": [
                "CACHE_MISS"
            ],
            "dependencyDigests": deps,
            "triggeringEventIds": [],
        }

    # Defensive fallback.
    return {
        "node": node,
        "action": "block",
        "reasonCodes": [
            "UPSTREAM_PENDING"
        ],
        "dependencyDigests": deps,
        "triggeringEventIds": [],
    }


# ============================================================
# REQUEST VALIDATION
# ============================================================

async def parse_request(request):

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

    if not required.issubset(body):
        raise ValueError("INVALID_REQUEST")

    if (
        not isinstance(body["session"], str)
        or not body["session"]
    ):
        raise ValueError("INVALID_REQUEST")

    if not positive_safe_integer(
        body["revision"]
    ):
        raise ValueError("INVALID_REQUEST")

    if not isinstance(body["inputs"], dict):
        raise ValueError("INVALID_REQUEST")

    if not isinstance(body["events"], list):
        raise ValueError("INVALID_REQUEST")

    # All 12 required inputs must be non-empty strings.
    for name in INPUT_NAMES:

        value = body["inputs"].get(name)

        if (
            not isinstance(value, str)
            or not value
        ):
            raise ValueError("INVALID_REQUEST")

    return body


# ============================================================
# POST /pipeline
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await parse_request(request)
    except ValueError as exc:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc)
            },
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    incoming_events = body["events"]

    # ========================================================
    # GET / CREATE BASE SESSION
    # ========================================================

    if session_id not in sessions:

        base = new_session(
            inputs,
            revision,
        )

    else:

        existing = sessions[session_id]

        # ----------------------------------------------------
        # SAME REVISION
        # ----------------------------------------------------

        if revision == existing["revision"]:

            if (
                compact_json(inputs)
                != existing["inputSignature"]
            ):
                return JSONResponse(
                    status_code=409,
                    content={
                        "error":
                            "REVISION_CONFLICT"
                    },
                )

            base = copy.deepcopy(existing)

        # ----------------------------------------------------
        # NEW REVISION
        # ----------------------------------------------------

        elif revision > existing["revision"]:

            base = copy.deepcopy(existing)

            # Replace current revision.
            base["revision"] = revision

            base["inputs"] = copy.deepcopy(
                inputs
            )

            base["inputSignature"] = (
                compact_json(inputs)
            )

            # Clear ALL current execution state.
            #
            # This includes successful current node state.
            # Successful results remain available through cache.
            base["nodes"] = {
                node: empty_node_state()
                for node in NODES
            }

            # KEEP:
            #
            # base["cache"]
            # base["evidence"]
            # base["events"]

        # ----------------------------------------------------
        # OLD REVISION
        # ----------------------------------------------------

        else:

            # Current revision remains authoritative.
            #
            # Events from the older revision will be ignored.
            base = copy.deepcopy(existing)

    # ========================================================
    # ATOMIC WORKING COPY
    # ========================================================

    working = copy.deepcopy(base)

    accepted = []
    ignored = []

    for event in incoming_events:

        # ----------------------------------------------------
        # STRUCTURE
        # ----------------------------------------------------

        if not event_structure_valid(event):

            return JSONResponse(
                status_code=409,
                content={
                    "error": "INVALID_EVENT"
                },
            )

        event_id = event["eventId"]

        canonical_event = compact_json(
            event
        )

        # ----------------------------------------------------
        # EVENT ID REPLAY / CONFLICT
        # ----------------------------------------------------

        if event_id in working["events"]:

            previous = working["events"][
                event_id
            ]

            if previous == canonical_event:

                ignored.append(event_id)
                continue

            return JSONResponse(
                status_code=409,
                content={
                    "error":
                        "EVENT_ID_CONFLICT"
                },
            )

        # ----------------------------------------------------
        # SEMANTICALLY INVALID EVENT
        #
        # Invalid status/artifact/receipt/attempt/node:
        # IGNORE. Do not consume event ID.
        # ----------------------------------------------------

        if not event_semantically_valid(event):

            ignored.append(event_id)
            continue

        # ----------------------------------------------------
        # WRONG REVISION
        #
        # Ignore and do not consume event ID.
        # ----------------------------------------------------

        if event["revision"] != working["revision"]:

            ignored.append(event_id)
            continue

        # ----------------------------------------------------
        # APPLY STATE TRANSITION
        # ----------------------------------------------------

        result = apply_event(
            working,
            event,
        )

        if result == "ignored":

            ignored.append(event_id)
            continue

        if result == "status_conflict":

            return JSONResponse(
                status_code=409,
                content={
                    "error":
                        "STATUS_CONFLICT"
                },
            )

        if result == "evidence_conflict":

            return JSONResponse(
                status_code=409,
                content={
                    "error":
                        "EVIDENCE_CONFLICT"
                },
            )

        # ----------------------------------------------------
        # ACCEPTED
        #
        # Only accepted events consume IDs.
        # ----------------------------------------------------

        working["events"][
            event_id
        ] = canonical_event

        accepted.append(event_id)

    # ========================================================
    # ATOMIC COMMIT
    # ========================================================

    sessions[session_id] = working

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "revision": working["revision"],
        "acceptedEventIds": accepted,
        "ignoredEventIds": ignored,
        "nodes": [
            make_node_response(
                working,
                node,
            )
            for node in NODES
        ],
    }


@app.get("/")
async def root():
    return {
        "status": "ok"
    }
