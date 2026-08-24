import hashlib
import json
import math
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

DB_FILE = "bqml_state.db"
DB_LOCK = threading.Lock()

SAFE_MAX = 9007199254740991

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ============================================================
# DATABASE
# ============================================================

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS selections (
                run_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
        """)
        conn.commit()


init_db()


# ============================================================
# GENERAL HELPERS
# ============================================================

def utf8(value):
    return value.encode("utf-8")


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False
    )


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False
    )


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_MAX
    )


def sorted_codes(codes):
    return sorted(set(codes), key=utf8)


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp(value):
    """
    Validate the exact timestamp grammar and return a UTC
    datetime with millisecond precision.
    """

    if not isinstance(value, str):
        return None

    m = TIME_RE.fullmatch(value)

    if not m:
        return None

    (
        ys, mos, ds,
        hs, mins, ss,
        fraction,
        zone
    ) = m.groups()

    year = int(ys)
    month = int(mos)
    day = int(ds)
    hour = int(hs)
    minute = int(mins)
    second = int(ss)

    if hour > 23:
        return None

    if minute > 59:
        return None

    if second > 59:
        return None

    if zone == "Z":
        tz = timezone.utc
    else:
        oh = int(zone[1:3])
        om = int(zone[4:6])

        if oh > 14:
            return None

        if om > 59:
            return None

        # +14:00 / -14:00 only.
        if oh == 14 and om != 0:
            return None

        sign = 1 if zone[0] == "+" else -1

        tz = timezone(
            sign * timedelta(
                hours=oh,
                minutes=om
            )
        )

    ms = 0

    if fraction:
        ms = int(fraction.ljust(3, "0"))

    try:
        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            ms * 1000,
            tzinfo=tz
        )
    except ValueError:
        return None

    return dt.astimezone(timezone.utc)


def utc_string(dt):
    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + "."
        + f"{dt.microsecond // 1000:03d}"
        + "Z"
    )


def normalize_timestamp(value):
    dt = parse_timestamp(value)

    if dt is None:
        return None

    return utc_string(dt)


# ============================================================
# FEATURE VALIDATION
# ============================================================

def valid_features(features):

    if not isinstance(features, dict):
        return False

    for name, feature in features.items():

        if not isinstance(name, str):
            return False

        if not isinstance(feature, dict):
            return False

        if set(feature.keys()) != {
            "value",
            "availableAt"
        }:
            return False

        if parse_timestamp(
            feature["availableAt"]
        ) is None:
            return False

    return True


# ============================================================
# SELECT ROW VALIDATION
# ============================================================

SELECT_KEYS = {
    "id",
    "entity",
    "eventTime",
    "predictionTime",
    "version",
    "split",
    "features"
}


def valid_select_row(row):

    if not isinstance(row, dict):
        return False

    if set(row.keys()) != SELECT_KEYS:
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if parse_timestamp(
        row["eventTime"]
    ) is None:
        return False

    if parse_timestamp(
        row["predictionTime"]
    ) is None:
        return False

    if not safe_int(row["version"]):
        return False

    if row["split"] not in {
        "TRAIN",
        "EVAL"
    }:
        return False

    if not valid_features(
        row["features"]
    ):
        return False

    return True


# ============================================================
# TRIAL VALIDATION
# ============================================================

TRIAL_KEYS = {
    "trialId",
    "status",
    "evalMetric"
}


def valid_trial(trial):

    if not isinstance(trial, dict):
        return False

    if set(trial.keys()) != TRIAL_KEYS:
        return False

    if not safe_int(
        trial["trialId"]
    ):
        return False

    if trial["status"] not in {
        "SUCCEEDED",
        "FAILED"
    }:
        return False

    metric = trial["evalMetric"]

    if trial["status"] == "SUCCEEDED":

        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(metric)
        ):
            return False

    else:

        if metric is not None:

            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(metric)
            ):
                return False

    return True


# ============================================================
# STATE
# ============================================================

def load_run(run_id):

    with DB_LOCK:

        with sqlite3.connect(DB_FILE) as conn:

            result = conn.execute(
                """
                SELECT request_json, response_json
                FROM selections
                WHERE run_id = ?
                """,
                (run_id,)
            ).fetchone()

    if result is None:
        return None

    return {
        "request_json": result[0],
        "response_json": result[1]
    }


def save_run(
    run_id,
    request_json,
    response_json
):

    with DB_LOCK:

        with sqlite3.connect(DB_FILE) as conn:

            conn.execute(
                """
                INSERT INTO selections
                (run_id, request_json, response_json)
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    request_json,
                    response_json
                )
            )

            conn.commit()


# ============================================================
# DATASET DIGEST
# ============================================================

def make_digest(
    train_ids,
    eval_ids,
    feature_names
):

    obj = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    data = compact_json(obj)

    return hashlib.sha256(
        data.encode("utf-8")
    ).hexdigest()


# ============================================================
# SELECT PHASE
# ============================================================

def select_phase(data):

    # --------------------------------------------------------
    # Exact top-level validation
    # --------------------------------------------------------

    required = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials"
    }

    if not isinstance(data, dict):
        return selection_error(
            None,
            ["INVALID_INPUT"]
        )

    if set(data.keys()) != required:
        return selection_error(
            data.get("runId"),
            ["INVALID_INPUT"]
        )

    if data["phase"] != "select":
        return selection_error(
            data.get("runId"),
            ["INVALID_INPUT"]
        )

    run_id = data["runId"]

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        return selection_error(
            None,
            ["INVALID_INPUT"]
        )

    forbidden = data[
        "forbiddenFeatures"
    ]

    if not isinstance(
        forbidden,
        list
    ):
        return selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    if any(
        not isinstance(x, str)
        for x in forbidden
    ):
        return selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    limit = data[
        "numTrialsLimit"
    ]

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > SAFE_MAX
    ):
        return selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    rows = data["rows"]
    trials = data["trials"]

    if not isinstance(rows, list):
        return selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    if not isinstance(trials, list):
        return selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    if len(rows) == 0:
        return selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    # --------------------------------------------------------
    # Validate all rows BEFORE processing them.
    # --------------------------------------------------------

    row_ids = set()

    for row in rows:

        if not valid_select_row(row):

            return selection_error(
                run_id,
                ["INVALID_INPUT"]
            )

        if row["id"] in row_ids:

            return selection_error(
                run_id,
                ["INVALID_INPUT"]
            )

        row_ids.add(
            row["id"]
        )

    # --------------------------------------------------------
    # Validate all trials.
    # --------------------------------------------------------

    trial_ids = set()

    for trial in trials:

        if not valid_trial(trial):

            return selection_error(
                run_id,
                ["INVALID_INPUT"]
            )

        if trial["trialId"] in trial_ids:

            return selection_error(
                run_id,
                ["INVALID_INPUT"]
            )

        trial_ids.add(
            trial["trialId"]
        )

    # --------------------------------------------------------
    # Trial count contract.
    # --------------------------------------------------------

    if len(trials) > limit:

        response = selection_error(
            run_id,
            ["TRIAL_LIMIT_EXCEEDED"]
        )

        return persist_selection(
            data,
            response
        )

    # --------------------------------------------------------
    # DEDUPLICATION
    #
    # Key = entity + UTC(eventTime)
    #
    # Highest version wins.
    # Equal version -> UTF-8 smallest ID.
    # --------------------------------------------------------

    groups = {}

    for row in rows:

        event_dt = parse_timestamp(
            row["eventTime"]
        )

        key = (
            row["entity"],
            event_dt
        )

        groups.setdefault(
            key,
            []
        ).append(row)

    retained = []

    for key, candidates in groups.items():

        winner = min(
            candidates,
            key=lambda r: (
                -r["version"],
                utf8(r["id"])
            )
        )

        retained.append(
            winner
        )

    # --------------------------------------------------------
    # SHARED FEATURE SET
    #
    # Feature must:
    # 1. occur in every retained row
    # 2. not be forbidden
    # 3. availableAt <= predictionTime
    #    for EVERY retained row
    # --------------------------------------------------------

    shared = None

    for row in retained:

        names = set(
            row["features"].keys()
        )

        if shared is None:
            shared = names
        else:
            shared &= names

    if shared is None:
        shared = set()

    forbidden_set = set(
        forbidden
    )

    eligible = []

    for name in shared:

        if name in forbidden_set:
            continue

        good = True

        for row in retained:

            available_dt = parse_timestamp(
                row["features"][name][
                    "availableAt"
                ]
            )

            prediction_dt = parse_timestamp(
                row["predictionTime"]
            )

            # Strict point-in-time condition.
            if available_dt > prediction_dt:
                good = False
                break

        if good:
            eligible.append(name)

    eligible.sort(key=utf8)

    # --------------------------------------------------------
    # SPLIT IDS
    # --------------------------------------------------------

    train_ids = [
        row["id"]
        for row in retained
        if row["split"] == "TRAIN"
    ]

    eval_ids = [
        row["id"]
        for row in retained
        if row["split"] == "EVAL"
    ]

    train_ids.sort(key=utf8)
    eval_ids.sort(key=utf8)

    # --------------------------------------------------------
    # TRIAL SELECTION
    # --------------------------------------------------------

    successful = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        metric = trial["evalMetric"]

        if (
            isinstance(metric, bool)
            or not isinstance(
                metric,
                (int, float)
            )
            or not math.isfinite(metric)
        ):
            continue

        successful.append(
            trial
        )

    if not successful:

        response = {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": eligible,
            "datasetDigest": None,
            "reasonCodes": [
                "NO_SUCCESSFUL_TRIAL"
            ]
        }

        return persist_selection(
            data,
            response
        )

    # Highest metric, smallest trialId tie.
    selected = sorted(
        successful,
        key=lambda t: (
            -t["evalMetric"],
            t["trialId"]
        )
    )[0]

    selected_trial_id = selected[
        "trialId"
    ]

    digest = make_digest(
        train_ids,
        eval_ids,
        eligible
    )

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": eligible,
        "datasetDigest": digest,
        "reasonCodes": []
    }

    return persist_selection(
        data,
        response
    )


# ============================================================
# PERSIST SELECTION
# ============================================================

def persist_selection(
    request_data,
    response
):

    run_id = response[
        "runId"
    ]

    request_json = canonical_json(
        request_data
    )

    existing = load_run(
        run_id
    )

    if existing is not None:

        if existing[
            "request_json"
        ] == request_json:

            return json.loads(
                existing["response_json"]
            )

        return JSONResponse(
            status_code=409,
            content={
                "error": "RUN_ID_CONFLICT"
            }
        )

    response_json = compact_json(
        response
    )

    save_run(
        run_id,
        request_json,
        response_json
    )

    return response


def selection_error(
    run_id,
    codes
):

    return {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": sorted_codes(codes)
    }


# ============================================================
# EVALUATION
# ============================================================

TEST_KEYS = {
    "label",
    "prediction",
    "slice"
}


def valid_test_row(row):

    if not isinstance(row, dict):
        return False

    if set(row.keys()) != TEST_KEYS:
        return False

    if (
        not isinstance(row["label"], int)
        or isinstance(row["label"], bool)
        or row["label"] not in (0, 1)
    ):
        return False

    if (
        not isinstance(row["prediction"], int)
        or isinstance(row["prediction"], bool)
        or row["prediction"] not in (0, 1)
    ):
        return False

    if (
        not isinstance(row["slice"], str)
        or not row["slice"]
    ):
        return False

    return True


def evaluate_phase(data):

    required = {
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes"
    }

    if not isinstance(data, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if set(data.keys()) != required:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if data["phase"] != "evaluate":
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    run_id = data["runId"]
    selected_trial = data[
        "selectedTrialId"
    ]
    digest = data[
        "datasetDigest"
    ]

    reason_codes = []

    # --------------------------------------------------------
    # Basic scalar validation
    # --------------------------------------------------------

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        reason_codes.append(
            "INVALID_INPUT"
        )

    if not safe_int(
        selected_trial
    ):
        reason_codes.append(
            "INVALID_INPUT"
        )

    if (
        not isinstance(digest, str)
        or HEX64_RE.fullmatch(
            digest
        ) is None
    ):
        reason_codes.append(
            "INVALID_INPUT"
        )

    floor = data["metricFloor"]

    if (
        isinstance(floor, bool)
        or not isinstance(
            floor,
            (int, float)
        )
        or not math.isfinite(floor)
        or floor < 0
        or floor > 1
    ):
        reason_codes.append(
            "INVALID_INPUT"
        )

    required_slices = data[
        "requiredSlices"
    ]

    if not isinstance(
        required_slices,
        dict
    ):
        reason_codes.append(
            "INVALID_INPUT"
        )
    else:

        for name, value in required_slices.items():

            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float)
                )
                or not math.isfinite(value)
                or value < 0
                or value > 1
            ):
                reason_codes.append(
                    "INVALID_INPUT"
                )
                break

    rows = data["rows"]

    if not isinstance(rows, list):
        reason_codes.append(
            "INVALID_INPUT"
        )

    bytes_processed = data[
        "bytesProcessed"
    ]

    max_bytes = data[
        "maxBytes"
    ]

    if not safe_int(bytes_processed):
        reason_codes.append(
            "INVALID_INPUT"
        )

    if not safe_int(max_bytes):
        reason_codes.append(
            "INVALID_INPUT"
        )

    if reason_codes:

        return evaluation_response(
            run_id,
            selected_trial,
            digest,
            None,
            False,
            bytes_processed,
            reason_codes
        )

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    stored = load_run(run_id)

    if stored is None:

        reason_codes.append(
            "INVALID_LINEAGE"
        )

    else:

        stored_response = json.loads(
            stored["response_json"]
        )

        if (
            stored_response[
                "selectedTrialId"
            ] != selected_trial
            or stored_response[
                "datasetDigest"
            ] != digest
        ):

            reason_codes.append(
                "INVALID_LINEAGE"
            )

    # --------------------------------------------------------
    # TEST ROWS
    # --------------------------------------------------------

    all_rows_valid = all(
        valid_test_row(row)
        for row in rows
    )

    if not all_rows_valid:

        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    test_metric = None
    critical_pass = False

    # --------------------------------------------------------
    # Empty / invalid test rows
    # --------------------------------------------------------

    if rows and all_rows_valid:

        correct = sum(
            row["label"]
            == row["prediction"]
            for row in rows
        )

        test_metric = round(
            correct / len(rows),
            12
        )

        if test_metric < floor:

            reason_codes.append(
                "AGGREGATE_FLOOR"
            )

        critical_pass = True

        present = {
            row["slice"]
            for row in rows
        }

        for name, required_floor in sorted(
            required_slices.items(),
            key=lambda x: utf8(x[0])
        ):

            if name not in present:

                reason_codes.append(
                    "MISSING_SLICE:" + name
                )

                critical_pass = False
                continue

            slice_rows = [
                row
                for row in rows
                if row["slice"] == name
            ]

            slice_correct = sum(
                row["label"]
                == row["prediction"]
                for row in slice_rows
            )

            slice_metric = round(
                slice_correct
                / len(slice_rows),
                12
            )

            if slice_metric < required_floor:

                reason_codes.append(
                    "SLICE_FLOOR:" + name
                )

                critical_pass = False

    # --------------------------------------------------------
    # Invalid lineage always fails criticalSlicePass.
    # --------------------------------------------------------

    if "INVALID_LINEAGE" in reason_codes:
        critical_pass = False

    # --------------------------------------------------------
    # Byte limit
    # --------------------------------------------------------

    if bytes_processed > max_bytes:

        reason_codes.append(
            "BYTE_LIMIT"
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    decision = (
        "admit"
        if not reason_codes
        else "reject"
    )

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": sorted_codes(
            reason_codes
        )
    }


def evaluation_response(
    run_id,
    selected_trial,
    digest,
    metric,
    critical,
    bytes_processed,
    codes
):

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "datasetDigest": digest,
        "testMetric": metric,
        "criticalSlicePass": critical,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": sorted_codes(codes)
    }


# ============================================================
# API
# ============================================================

@app.post("/bqml")
async def bqml(request: Request):

    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if not isinstance(data, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    phase = data.get("phase")

    if phase == "select":
        return select_phase(data)

    if phase == "evaluate":
        return evaluate_phase(data)

    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_INPUT"
        }
    )


@app.get("/")
def root():

    return {
        "status": "ok",
        "endpoint": "/bqml"
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )