import copy
import hashlib
import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

NODES = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

NODE_INDEX = {n: i for i, n in enumerate(NODES)}

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

# Process-level persistence across requests.
# Every session owns completely separate state.
sessions = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_array(values):
    """
    SHA-256 over UTF-8 compact JSON array.

    DO NOT sort this array.
    The order here is the DAG-specified order.
    """
    raw = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def digest_string(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_positive_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= 9007199254740991
    )


def empty_node():
    return {
        "status": None,
        "attempt": None,
        "artifactDigest": None,
        "startEventId": None,
        "retryEventId": None,
        "successEventId": None,
        "terminalEventId": None,
    }


def parent_of(node):
    i = NODE_INDEX[node]

    if i == 0:
        return None

    return NODES[i - 1]


def cache_id(node, key):
    return (node, key)


# ============================================================
# SESSION
# ============================================================

def make_session(inputs, revision):
    return {
        "revision": revision,

        # Preserve the COMPLETE inputs object.
        # This catches conflicts caused by extra metadata too.
        "inputs": copy.deepcopy(inputs),

        # Canonical representation used for exact revision comparison.
        "inputSignature": compact_json(inputs),

        # Current revision execution state.
        "nodes": {
            node: empty_node()
            for node in NODES
        },

        # Successful content-addressed results survive revisions.
        #
        # (node, key) ->
        # {
        #     "artifactDigest": ...,
        #     "eventId": ...
        # }
        "cache": {},

        # Immutable first-success binding.
        #
        # (node, key) ->
        # {
        #     "artifactDigest": ...,
        #     "eventId": ...
        # }
        "evidence": {},

        # eventId -> canonical compact event JSON
        "events": {},
    }


# ============================================================
# EXACT DAG DEPENDENCY ARRAYS
# ============================================================

def dependency_array(session, node):
    """
    These arrays MUST remain in the exact order from the prompt.

    Parent artifact values are obtained through reusable_artifact(),
    meaning either current-revision success OR successful cache.
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
        return [
            reusable_artifact(session, "prepare"),
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    if node == "evaluate":
        return [
            reusable_artifact(session, "train"),
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    if node == "register":
        return [
            reusable_artifact(session, "evaluate"),
            inputs["schemaDigest"],
        ]

    if node == "publish":
        return [
            reusable_artifact(session, "register"),
            inputs["publishConfig"],
        ]

    raise ValueError(node)


def compute_key(session, node):
    deps = dependency_array(session, node)

    # Parent-gated key:
    # if any parent artifact is unavailable, downstream key is null.
    if any(value is None for value in deps):
        return None

    return sha256_array(deps)


# ============================================================
# CACHE / REUSABILITY
# ============================================================

def reusable_artifact(session, node):
    """
    Return the artifact that makes this node reusable.

    Priority:
      1. Current revision successful state
      2. Successful content-addressed cache

    For downstream nodes this function recursively allows a cached
    parent to unlock the parent's descendants.
    """

    state = session["nodes"][node]

    if state["status"] == "succeeded":
        return state["artifactDigest"]

    # To determine this node's cache key, its own parents must first
    # be reusable.
    key = compute_key(session, node)

    if key is None:
        return None

    cached = session["cache"].get(cache_id(node, key))

    if cached is None:
        return None

    return cached["artifactDigest"]


def cache_hit(session, node):
    key = compute_key(session, node)

    if key is None:
        return None

    return session["cache"].get(cache_id(node, key))


# ============================================================
# RESPONSE DEPENDENCY DIGESTS
# ============================================================

def dependency_digest_map(session, node, key):
    inputs = session["inputs"]

    result = {}

    if node == "verify_data":
        result["generation"] = digest_string(inputs["generation"])
        result["checksum"] = digest_string(inputs["checksum"])

    elif node == "prepare":
        result["canonicalData"] = digest_string(
            inputs["canonicalData"]
        )
        result["prepareCode"] = digest_string(
            inputs["prepareCode"]
        )
        result["prepareConfig"] = digest_string(
            inputs["prepareConfig"]
        )

    elif node == "train":
        result["prepareArtifact"] = reusable_artifact(
            session,
            "prepare",
        )
        result["trainCode"] = digest_string(
            inputs["trainCode"]
        )
        result["trainConfig"] = digest_string(
            inputs["trainConfig"]
        )
        result["runtime"] = digest_string(
            inputs["runtime"]
        )

    elif node == "evaluate":
        result["trainArtifact"] = reusable_artifact(
            session,
            "train",
        )
        result["canonicalData"] = digest_string(
            inputs["canonicalData"]
        )
        result["evaluateCode"] = digest_string(
            inputs["evaluateCode"]
        )
        result["evaluateConfig"] = digest_string(
            inputs["evaluateConfig"]
        )

    elif node == "register":
        result["evaluateArtifact"] = reusable_artifact(
            session,
            "evaluate",
        )
        result["schemaDigest"] = digest_string(
            inputs["schemaDigest"]
        )

    elif node == "publish":
        result["registerArtifact"] = reusable_artifact(
            session,
            "register",
        )
        result["publishConfig"] = digest_string(
            inputs["publishConfig"]
        )

    result["cacheKey"] = key

    return result


# ============================================================
# EVENT VALIDATION
# ============================================================

def valid_event_shape(event):
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

    # EXACTLY eight fields.
    if set(event.keys()) != expected:
        return False

    if not isinstance(event["eventId"], str) or not event["eventId"]:
        return False

    if not safe_positive_int(event["revision"]):
        return False

    if event["node"] not in NODE_INDEX:
        return False

    if not safe_positive_int(event["attempt"]):
        return False

    if event["status"] not in STATUSES:
        return False

    if not isinstance(event["key"], str) or not event["key"]:
        return False

    # Success requires artifact.
    if event["status"] == "succeeded":
        if (
            not isinstance(event["artifactDigest"], str)
            or not event["artifactDigest"]
        ):
            return False

    # Everything else requires null artifact.
    else:
        if event["artifactDigest"] is not None:
            return False

    # Register/publish successful completion requires receipt.
    if (
        event["node"] in ("register", "publish")
        and event["status"] == "succeeded"
    ):
        expected_receipt = (
            f"receipt:{event['node']}:{event['key']}"
        )

        if event["receiptId"] != expected_receipt:
            return False

    else:
        if event["receiptId"] is not None:
            return False

    return True


# ============================================================
# EVENT TRANSITIONS
# ============================================================

def process_event(session, event):
    node = event["node"]
    state = session["nodes"][node]

    # Wrong revision is ignored.
    if event["revision"] != session["revision"]:
        return "ignored"

    current_key = compute_key(session, node)

    # Parent unavailable.
    if current_key is None:
        return "ignored"

    # Wrong key.
    if event["key"] != current_key:
        return "ignored"

    cid = cache_id(node, current_key)

    # --------------------------------------------------------
    # Immutable successful evidence.
    # --------------------------------------------------------

    evidence = session["evidence"].get(cid)

    if evidence is not None:

        if event["status"] == "succeeded":

            if (
                event["artifactDigest"]
                != evidence["artifactDigest"]
            ):
                return "evidence_conflict"

        # Once a key has succeeded, any new event is a status
        # conflict, except an exact event replay which was handled
        # before this function.
        return "status_conflict"

    # --------------------------------------------------------
    # No current state.
    # --------------------------------------------------------

    if state["status"] is None:

        # Only started(1) can initiate a node.
        if (
            event["status"] == "started"
            and event["attempt"] == 1
        ):
            state["status"] = "started"
            state["attempt"] = 1
            state["startEventId"] = event["eventId"]

            return "accepted"

        # Completion / attempt > 1 without initial start.
        return "ignored"

    # --------------------------------------------------------
    # started(n)
    # --------------------------------------------------------

    if state["status"] == "started":

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
                state["artifactDigest"] = (
                    event["artifactDigest"]
                )
                state["successEventId"] = event["eventId"]

                # Permanently bind the first success.
                session["evidence"][cid] = {
                    "artifactDigest": (
                        event["artifactDigest"]
                    ),
                    "eventId": event["eventId"],
                }

                session["cache"][cid] = {
                    "artifactDigest": (
                        event["artifactDigest"]
                    ),
                    "eventId": event["eventId"],
                }

            elif event["status"] == "retryable_failed":

                state["status"] = "retryable_failed"
                state["retryEventId"] = event["eventId"]

            else:

                state["status"] = "terminal_failed"
                state["terminalEventId"] = (
                    event["eventId"]
                )

            return "accepted"

        # Lower attempt is ignored.
        if event["attempt"] < state["attempt"]:
            return "ignored"

        return "status_conflict"

    # --------------------------------------------------------
    # retryable_failed(n)
    # --------------------------------------------------------

    if state["status"] == "retryable_failed":

        if (
            event["status"] == "started"
            and event["attempt"] == state["attempt"] + 1
        ):
            state["status"] = "started"
            state["attempt"] = event["attempt"]
            state["startEventId"] = event["eventId"]

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
# NODE RESPONSE
# ============================================================

def node_response(session, node):
    state = session["nodes"][node]

    key = compute_key(session, node)

    deps = (
        dependency_digest_map(session, node, key)
        if key is not None
        else {}
    )

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    cached = cache_hit(session, node)

    if cached is not None:

        return {
            "node": node,
            "action": "reuse",
            "reasonCodes": ["CACHE_HIT"],
            "dependencyDigests": deps,
            "triggeringEventIds": [
                cached["eventId"]
            ],
        }

    # --------------------------------------------------------
    # Parent status
    # --------------------------------------------------------

    parent = parent_of(node)

    if parent is not None:

        parent_state = session["nodes"][parent]

        # Parent terminal failure.
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

        # Parent is not reusable.
        if reusable_artifact(session, parent) is None:

            triggering = []

            if parent_state["startEventId"]:
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
    # Current node terminal.
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
    # Current node running.
    # --------------------------------------------------------

    if state["status"] == "started":

        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["RUNNING"],
            "dependencyDigests": deps,
            "triggeringEventIds": [
                state["startEventId"]
            ],
        }

    # --------------------------------------------------------
    # Retryable failure.
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
    # Ready but uncached.
    # --------------------------------------------------------

    if key is not None:

        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": ["CACHE_MISS"],
            "dependencyDigests": deps,
            "triggeringEventIds": [],
        }

    # Should only happen when parent isn't reusable.
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

async def read_request(request: Request):

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

    if (
        not isinstance(session_id, str)
        or not session_id
    ):
        raise ValueError("INVALID_REQUEST")

    if not safe_positive_int(body["revision"]):
        raise ValueError("INVALID_REQUEST")

    if not isinstance(body["inputs"], dict):
        raise ValueError("INVALID_REQUEST")

    if not isinstance(body["events"], list):
        raise ValueError("INVALID_REQUEST")

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
        body = await read_request(request)
    except ValueError as exc:
        return JSONResponse(
            status_code=409,
            content={"error": str(exc)},
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # ========================================================
    # DETERMINE BASE STATE
    # ========================================================

    if session_id not in sessions:

        # Brand-new session.
        base = make_session(
            inputs,
            revision,
        )

    else:

        existing = sessions[session_id]

        # ----------------------------------------------------
        # Same revision
        # ----------------------------------------------------

        if revision == existing["revision"]:

            # Inputs must be byte-for-byte equivalent under
            # our compact canonical JSON representation.
            #
            # This includes EXTRA metadata as well.
            if (
                compact_json(inputs)
                != existing["inputSignature"]
            ):
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "REVISION_CONFLICT"
                    },
                )

            # Work from existing state.
            base = copy.deepcopy(existing)

        # ----------------------------------------------------
        # Newer revision
        # ----------------------------------------------------

        elif revision > existing["revision"]:

            # Start with existing session so successful cache
            # and immutable evidence survive.
            base = copy.deepcopy(existing)

            # Replace revision inputs.
            base["revision"] = revision
            base["inputs"] = copy.deepcopy(inputs)
            base["inputSignature"] = compact_json(inputs)

            # New revision gets fresh execution state.
            base["nodes"] = {
                node: empty_node()
                for node in NODES
            }

            # DO NOT clear:
            #
            # base["cache"]
            # base["evidence"]
            # base["events"]
            #
            # Successful cache/evidence survives revisions.

        # ----------------------------------------------------
        # Older revision
        # ----------------------------------------------------

        else:

            # Keep current session state.
            #
            # Events belonging to an older revision are ignored
            # later and do not consume their event IDs.
            base = copy.deepcopy(existing)

    # ========================================================
    # ATOMIC EVENT PROCESSING
    # ========================================================
    #
    # Everything happens against "working".
    #
    # If ANY event causes a 409, we return immediately and
    # sessions[session_id] is never modified.
    #

    working = copy.deepcopy(base)

    accepted_ids = []
    ignored_ids = []

    for event in events:

        # ----------------------------------------------------
        # Validate event structure.
        # ----------------------------------------------------

        if not valid_event_shape(event):
            return JSONResponse(
                status_code=409,
                content={
                    "error": "INVALID_EVENT"
                },
            )

        event_id = event["eventId"]
        canonical_event = compact_json(event)

        # ----------------------------------------------------
        # Global event ID handling.
        # ----------------------------------------------------
        #
        # Exact replay:
        #   ignore
        #
        # Same ID + different content:
        #   409 EVENT_ID_CONFLICT
        #

        if event_id in working["events"]:

            if (
                working["events"][event_id]
                == canonical_event
            ):
                ignored_ids.append(event_id)
                continue

            return JSONResponse(
                status_code=409,
                content={
                    "error": "EVENT_ID_CONFLICT"
                },
            )

        # ----------------------------------------------------
        # Older/wrong revision.
        #
        # IMPORTANT:
        # Do NOT put this event into working["events"].
        # Therefore it does not consume its ID.
        # ----------------------------------------------------

        if event["revision"] != working["revision"]:

            ignored_ids.append(event_id)
            continue

        # ----------------------------------------------------
        # State-machine transition.
        # ----------------------------------------------------

        result = process_event(
            working,
            event,
        )

        # ----------------------------------------------------
        # Ignored event.
        # ----------------------------------------------------

        if result == "ignored":

            ignored_ids.append(event_id)

            # Ignored IDs are NOT consumed.
            continue

        # ----------------------------------------------------
        # Status conflict.
        # ----------------------------------------------------

        if result == "status_conflict":

            return JSONResponse(
                status_code=409,
                content={
                    "error": "STATUS_CONFLICT"
                },
            )

        # ----------------------------------------------------
        # Immutable evidence conflict.
        # ----------------------------------------------------

        if result == "evidence_conflict":

            return JSONResponse(
                status_code=409,
                content={
                    "error": "EVIDENCE_CONFLICT"
                },
            )

        # ----------------------------------------------------
        # Accepted event.
        #
        # Only accepted events consume their IDs.
        # ----------------------------------------------------

        working["events"][event_id] = canonical_event
        accepted_ids.append(event_id)

    # ========================================================
    # ATOMIC COMMIT
    # ========================================================

    sessions[session_id] = working

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "revision": working["revision"],
        "acceptedEventIds": accepted_ids,
        "ignoredEventIds": ignored_ids,
        "nodes": [
            node_response(working, node)
            for node in NODES
        ],
    }


@app.get("/")
async def root():
    return {"status": "ok"}
