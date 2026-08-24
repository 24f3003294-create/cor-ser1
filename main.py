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


# ============================================================
# CONSTANTS
# ============================================================

SAFE_INT_MAX = 9007199254740991

TIME_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)

HEX64_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


# ============================================================
# STATE
# ============================================================

DB_FILE = "bqml_state.db"

db_lock = threading.Lock()


def init_db():

    with sqlite3.connect(DB_FILE) as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS selections (
                run_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
            """
        )

        conn.commit()


init_db()


# ============================================================
# BASIC HELPERS
# ============================================================

def utf8_key(value):
    return value.encode("utf-8")


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False
    )


def canonical_request(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False
    )


def sorted_unique_codes(codes):

    return sorted(
        set(codes),
        key=utf8_key
    )


# ============================================================
# SAFE INTEGER
# ============================================================

def is_safe_integer(value):

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


# ============================================================
# TIME VALIDATION / UTC NORMALIZATION
# ============================================================

def normalize_time(value):

    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if match is None:
        return None

    (
        year,
        month,
        day,
        hour,
        minute,
        second,
        fraction,
        offset
    ) = match.groups()

    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    if hour > 23:
        return None

    if minute > 59:
        return None

    if second > 59:
        return None

    # --------------------------------------------------------
    # UTC
    # --------------------------------------------------------

    if offset == "Z":

        tz = timezone.utc

    else:

        offset_hour = int(
            offset[1:3]
        )

        offset_minute = int(
            offset[4:6]
        )

        if offset_hour > 14:
            return None

        if offset_minute > 59:
            return None

        # +14:00 and -14:00 are valid,
        # but +14:01 is not.
        if (
            offset_hour == 14
            and offset_minute != 0
        ):
            return None

        sign = (
            1
            if offset[0] == "+"
            else -1
        )

        tz = timezone(
            sign * timedelta(
                hours=offset_hour,
                minutes=offset_minute
            )
        )

    milliseconds = 0

    if fraction is not None:

        milliseconds = int(
            fraction.ljust(3, "0")
        )

    try:

        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            milliseconds * 1000,
            tzinfo=tz
        )

    except ValueError:

        return None

    dt = dt.astimezone(
        timezone.utc
    )

    return (
        dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + "."
        + f"{dt.microsecond // 1000:03d}"
        + "Z"
    )


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature_map(features):

    if not isinstance(
        features,
        dict
    ):
        return False

    for name, feature in features.items():

        if not isinstance(
            name,
            str
        ):
            return False

        if not name:
            return False

        if not isinstance(
            feature,
            dict
        ):
            return False

        if set(feature.keys()) != {
            "value",
            "availableAt"
        }:
            return False

        if normalize_time(
            feature["availableAt"]
        ) is None:
            return False

    return True


# ============================================================
# SELECTION ROW VALIDATION
# ============================================================

REQUIRED_SELECT_ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "predictionTime",
    "version",
    "split",
    "features"
}


def validate_selection_row(row):

    if not isinstance(
        row,
        dict
    ):
        return False

    if set(row.keys()) != REQUIRED_SELECT_ROW_KEYS:
        return False

    if not isinstance(
        row["id"],
        str
    ) or not row["id"]:

        return False

    if not isinstance(
        row["entity"],
        str
    ):

        return False

    if normalize_time(
        row["eventTime"]
    ) is None:

        return False

    if normalize_time(
        row["predictionTime"]
    ) is None:

        return False

    if not is_safe_integer(
        row["version"]
    ):

        return False

    if row["split"] not in {
        "TRAIN",
        "EVAL"
    }:

        return False

    if not validate_feature_map(
        row["features"]
    ):

        return False

    return True


# ============================================================
# TRIAL VALIDATION
# ============================================================

REQUIRED_TRIAL_KEYS = {
    "trialId",
    "status",
    "evalMetric"
}


def validate_trial(trial):

    if not isinstance(
        trial,
        dict
    ):

        return False

    if set(trial.keys()) != REQUIRED_TRIAL_KEYS:
        return False

    if not is_safe_integer(
        trial["trialId"]
    ):

        return False

    if trial["status"] not in {
        "SUCCEEDED",
        "FAILED"
    }:

        return False

    metric = trial["evalMetric"]

    # FAILED trials may have null metrics.
    if trial["status"] == "FAILED":

        if metric is not None:

            if (
                isinstance(metric, bool)
                or not isinstance(
                    metric,
                    (int, float)
                )
                or not math.isfinite(metric)
            ):

                return False

    else:

        # SUCCEEDED needs a finite metric.
        if (
            isinstance(metric, bool)
            or not isinstance(
                metric,
                (int, float)
            )
            or not math.isfinite(metric)
        ):

            return False

    return True


# ============================================================
# GET STORED SELECTION
# ============================================================

def get_selection(run_id):

    with db_lock:

        with sqlite3.connect(DB_FILE) as conn:

            row = conn.execute(
                """
                SELECT request_json, response_json
                FROM selections
                WHERE run_id = ?
                """,
                (run_id,)
            ).fetchone()

    if row is None:
        return None

    return {
        "request_json": row[0],
        "response_json": row[1]
    }


# ============================================================
# SAVE SELECTION
# ============================================================

def save_selection(
    run_id,
    request_json,
    response_json
):

    with db_lock:

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
# BUILD DATASET DIGEST
# ============================================================

def dataset_digest(
    train_ids,
    eval_ids,
    feature_names
):

    payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    exact_json = compact_json(
        payload
    )

    return hashlib.sha256(
        exact_json.encode("utf-8")
    ).hexdigest()


# ============================================================
# SELECT PHASE
# ============================================================

def select_phase(data):

    reason_codes = []

    # --------------------------------------------------------
    # Top-level validation
    # --------------------------------------------------------

    if not isinstance(
        data,
        dict
    ):

        return make_selection_error(
            None,
            ["INVALID_INPUT"]
        )

    required = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials"
    }

    if set(data.keys()) != required:

        return make_selection_error(
            data.get("runId"),
            ["INVALID_INPUT"]
        )

    if data["phase"] != "select":

        return make_selection_error(
            data.get("runId"),
            ["INVALID_INPUT"]
        )

    run_id = data["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):

        return make_selection_error(
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

        return make_selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    if any(
        not isinstance(x, str)
        for x in forbidden
    ):

        return make_selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    num_limit = data[
        "numTrialsLimit"
    ]

    if (
        not isinstance(num_limit, int)
        or isinstance(num_limit, bool)
        or num_limit <= 0
        or num_limit > SAFE_INT_MAX
    ):

        return make_selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    rows = data["rows"]
    trials = data["trials"]

    if not isinstance(rows, list):
        return make_selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    if not isinstance(trials, list):
        return make_selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    # Selection rows must be non-empty.
    if len(rows) == 0:

        return make_selection_error(
            run_id,
            ["INVALID_INPUT"]
        )

    # --------------------------------------------------------
    # Trial limit
    # --------------------------------------------------------

    if len(trials) > num_limit:

        reason_codes.append(
            "TRIAL_LIMIT_EXCEEDED"
        )

    # --------------------------------------------------------
    # Validate rows
    # --------------------------------------------------------

    row_ids = set()

    valid_rows = True

    for row in rows:

        if not validate_selection_row(
            row
        ):

            valid_rows = False
            break

        if row["id"] in row_ids:

            valid_rows = False
            break

        row_ids.add(
            row["id"]
        )

    if not valid_rows:

        reason_codes.append(
            "INVALID_INPUT"
        )

    # --------------------------------------------------------
    # Validate trials
    # --------------------------------------------------------

    trial_ids = set()

    valid_trials = True

    for trial in trials:

        if not validate_trial(
            trial
        ):

            valid_trials = False
            break

        if trial["trialId"] in trial_ids:

            valid_trials = False
            break

        trial_ids.add(
            trial["trialId"]
        )

    if not valid_trials:

        reason_codes.append(
            "INVALID_INPUT"
        )

    # --------------------------------------------------------
    # If malformed, return immediately.
    # --------------------------------------------------------

    if "INVALID_INPUT" in reason_codes:

        return make_selection_error(
            run_id,
            reason_codes
        )

    # --------------------------------------------------------
    # Deduplicate rows
    #
    # Key:
    # [entity, UTC(eventTime)]
    #
    # Highest version wins.
    # Equal version -> UTF-8 smallest ID.
    # --------------------------------------------------------

    groups = {}

    for row in rows:

        utc_event = normalize_time(
            row["eventTime"]
        )

        key = (
            row["entity"],
            utc_event
        )

        groups.setdefault(
            key,
            []
        ).append(row)

    retained_rows = []

    for group in groups.values():

        winner = sorted(
            group,
            key=lambda row: (
                -row["version"],
                utf8_key(row["id"])
            )
        )[0]

        retained_rows.append(
            winner
        )

    # --------------------------------------------------------
    # Determine eligible features
    # --------------------------------------------------------

    forbidden_set = set(
        forbidden
    )

    # Feature must appear in every retained row.
    common_features = None

    for row in retained_rows:

        names = set(
            row["features"].keys()
        )

        if common_features is None:

            common_features = names

        else:

            common_features &= names

    if common_features is None:

        common_features = set()

    feature_names = []

    for name in common_features:

        if name in forbidden_set:
            continue

        eligible = True

        for row in retained_rows:

            feature = row[
                "features"
            ][name]

            available_at = normalize_time(
                feature["availableAt"]
            )

            prediction_time = normalize_time(
                row["predictionTime"]
            )

            if (
                available_at
                > prediction_time
            ):

                eligible = False
                break

        if eligible:

            feature_names.append(
                name
            )

    feature_names.sort(
        key=utf8_key
    )

    # --------------------------------------------------------
    # TRAIN / EVAL IDs
    # --------------------------------------------------------

    train_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "TRAIN"
        ],
        key=utf8_key
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "EVAL"
        ],
        key=utf8_key
    )

    # --------------------------------------------------------
    # Select successful trial
    # --------------------------------------------------------

    successful_trials = [
        trial
        for trial in trials
        if (
            trial["status"] == "SUCCEEDED"
            and isinstance(
                trial["evalMetric"],
                (int, float)
            )
            and not isinstance(
                trial["evalMetric"],
                bool
            )
            and math.isfinite(
                trial["evalMetric"]
            )
        )
    ]

    if not successful_trials:

        reason_codes.append(
            "NO_SUCCESSFUL_TRIAL"
        )

    # --------------------------------------------------------
    # Select maximum metric.
    #
    # Exact tie -> smallest integer trialId.
    # --------------------------------------------------------

    selected_trial_id = None

    if successful_trials:

        winner = sorted(
            successful_trials,
            key=lambda trial: (
                -trial["evalMetric"],
                trial["trialId"]
            )
        )[0]

        selected_trial_id = winner[
            "trialId"
        ]

    # --------------------------------------------------------
    # Digest
    # --------------------------------------------------------

    digest = dataset_digest(
        train_ids,
        eval_ids,
        feature_names
    )

    # If any reason code exists,
    # selection is unsuccessful.
    if reason_codes:

        selected_trial_id = None

        if "INVALID_INPUT" in reason_codes:

            digest = None

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": sorted_unique_codes(
            reason_codes
        )
    }

    # --------------------------------------------------------
    # Stateful persistence
    # --------------------------------------------------------

    request_json = canonical_request(
        data
    )

    existing = get_selection(
        run_id
    )

    if existing is not None:

        if (
            existing["request_json"]
            == request_json
        ):

            # Exact stored response.
            return json.loads(
                existing["response_json"]
            )

        return JSONResponse(
            status_code=409,
            content={
                "error": "RUN_ID_CONFLICT"
            }
        )

    # Persist complete response.
    save_selection(
        run_id,
        request_json,
        compact_json(response)
    )

    return response


# ============================================================
# SELECTION ERROR
# ============================================================

def make_selection_error(
    run_id,
    codes
):

    response = {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": sorted_unique_codes(
            codes
        )
    }

    return response


# ============================================================
# EVALUATION ROW VALIDATION
# ============================================================

REQUIRED_TEST_ROW_KEYS = {
    "label",
    "prediction",
    "slice"
}


def validate_test_row(row):

    if not isinstance(
        row,
        dict
    ):

        return False

    if set(row.keys()) != REQUIRED_TEST_ROW_KEYS:

        return False

    label = row["label"]
    prediction = row["prediction"]
    slice_name = row["slice"]

    if (
        not isinstance(label, int)
        or isinstance(label, bool)
        or label not in (0, 1)
    ):

        return False

    if (
        not isinstance(prediction, int)
        or isinstance(prediction, bool)
        or prediction not in (0, 1)
    ):

        return False

    if (
        not isinstance(slice_name, str)
        or not slice_name
    ):

        return False

    return True


# ============================================================
# EVALUATION PHASE
# ============================================================

def evaluate_phase(data):

    reason_codes = []

    # --------------------------------------------------------
    # Basic input
    # --------------------------------------------------------

    if not isinstance(
        data,
        dict
    ):

        return make_evaluation_response(
            None,
            None,
            None,
            None,
            False,
            0,
            ["INVALID_INPUT"]
        )

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

    if set(data.keys()) != required:

        return make_evaluation_response(
            data.get("runId"),
            data.get("selectedTrialId"),
            data.get("datasetDigest"),
            None,
            False,
            data.get("bytesProcessed"),
            ["INVALID_INPUT"]
        )

    if data["phase"] != "evaluate":

        return make_evaluation_response(
            data.get("runId"),
            data.get("selectedTrialId"),
            data.get("datasetDigest"),
            None,
            False,
            data.get("bytesProcessed"),
            ["INVALID_INPUT"]
        )

    run_id = data["runId"]
    selected_trial_id = data[
        "selectedTrialId"
    ]
    supplied_digest = data[
        "datasetDigest"
    ]

    # --------------------------------------------------------
    # Validate scalar inputs
    # --------------------------------------------------------

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):

        reason_codes.append(
            "INVALID_INPUT"
        )

    if (
        not is_safe_integer(
            selected_trial_id
        )
    ):

        reason_codes.append(
            "INVALID_INPUT"
        )

    if (
        not isinstance(
            supplied_digest,
            str
        )
        or HEX64_RE.fullmatch(
            supplied_digest
        ) is None
    ):

        reason_codes.append(
            "INVALID_INPUT"
        )

    metric_floor = data[
        "metricFloor"
    ]

    if (
        isinstance(metric_floor, bool)
        or not isinstance(
            metric_floor,
            (int, float)
        )
        or not math.isfinite(
            metric_floor
        )
        or metric_floor < 0
        or metric_floor > 1
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

        for name, floor in required_slices.items():

            if (
                not isinstance(name, str)
                or not name
                or isinstance(floor, bool)
                or not isinstance(
                    floor,
                    (int, float)
                )
                or not math.isfinite(
                    floor
                )
                or floor < 0
                or floor > 1
            ):

                reason_codes.append(
                    "INVALID_INPUT"
                )

                break

    bytes_processed = data[
        "bytesProcessed"
    ]

    max_bytes = data[
        "maxBytes"
    ]

    if not is_safe_integer(
        bytes_processed
    ):

        reason_codes.append(
            "INVALID_INPUT"
        )

    if not is_safe_integer(
        max_bytes
    ):

        reason_codes.append(
            "INVALID_INPUT"
        )

    rows = data["rows"]

    if not isinstance(
        rows,
        list
    ):

        reason_codes.append(
            "INVALID_INPUT"
        )

    # --------------------------------------------------------
    # If basic input is invalid
    # --------------------------------------------------------

    if "INVALID_INPUT" in reason_codes:

        return make_evaluation_response(
            run_id,
            selected_trial_id,
            supplied_digest,
            None,
            False,
            bytes_processed,
            reason_codes
        )

    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    stored = get_selection(
        run_id
    )

    if stored is None:

        reason_codes.append(
            "INVALID_LINEAGE"
        )

    else:

        stored_response = json.loads(
            stored["response_json"]
        )

        stored_trial = stored_response[
            "selectedTrialId"
        ]

        stored_digest = stored_response[
            "datasetDigest"
        ]

        # Must be a successful selection.
        if (
            stored_trial is None
            or stored_digest is None
            or stored_trial
            != selected_trial_id
            or stored_digest
            != supplied_digest
        ):

            reason_codes.append(
                "INVALID_LINEAGE"
            )

    # --------------------------------------------------------
    # Test rows
    # --------------------------------------------------------

    rows_valid = True

    for row in rows:

        if not validate_test_row(
            row
        ):

            rows_valid = False
            break

    if not rows_valid:

        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    # --------------------------------------------------------
    # Test metric
    #
    # If rows are empty or invalid:
    # testMetric = null
    # Skip aggregate and slice checks.
    # --------------------------------------------------------

    test_metric = None
    critical_slice_pass = True

    if len(rows) == 0 or not rows_valid:

        test_metric = None
        critical_slice_pass = False

    else:

        # ----------------------------------------------------
        # Aggregate accuracy
        # ----------------------------------------------------

        correct = sum(
            1
            for row in rows
            if row["label"]
            == row["prediction"]
        )

        test_metric = round(
            correct / len(rows),
            12
        )

        # ----------------------------------------------------
        # Aggregate floor
        # ----------------------------------------------------

        if test_metric < metric_floor:

            reason_codes.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # Required slices
        # ----------------------------------------------------

        present_slices = {
            row["slice"]
            for row in rows
        }

        for slice_name, floor in sorted(
            required_slices.items(),
            key=lambda item:
                utf8_key(item[0])
        ):

            if slice_name not in present_slices:

                reason_codes.append(
                    "MISSING SLICE:"
                    + slice_name
                )

                critical_slice_pass = False

                continue

            slice_rows = [
                row
                for row in rows
                if row["slice"]
                == slice_name
            ]

            slice_correct = sum(
                1
                for row in slice_rows
                if row["label"]
                == row["prediction"]
            )

            slice_metric = round(
                slice_correct
                / len(slice_rows),
                12
            )

            if slice_metric < floor:

                reason_codes.append(
                    "SLICE_FLOOR:"
                    + slice_name
                )

                critical_slice_pass = False

    # Invalid test rows already make criticalSlicePass false.
    if not rows_valid:

        critical_slice_pass = False

    # Invalid lineage also makes criticalSlicePass false.
    if "INVALID_LINEAGE" in reason_codes:

        critical_slice_pass = False

    # --------------------------------------------------------
    # Byte limit
    # --------------------------------------------------------

    if bytes_processed > max_bytes:

        reason_codes.append(
            "BYTE_LIMIT"
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if reason_codes:

        decision = "reject"

    else:

        decision = "admit"

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": supplied_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": sorted_unique_codes(
            reason_codes
        )
    }

    return response


# ============================================================
# EVALUATION RESPONSE
# ============================================================

def make_evaluation_response(
    run_id,
    selected_trial_id,
    digest,
    test_metric,
    critical_slice_pass,
    bytes_processed,
    reason_codes
):

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": sorted_unique_codes(
            reason_codes
        )
    }


# ============================================================
# MAIN ENDPOINT
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

    if not isinstance(
        data,
        dict
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    phase = data.get(
        "phase"
    )

    # Unknown or missing phase -> HTTP 400
    if phase not in {
        "select",
        "evaluate"
    }:

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if phase == "select":

        result = select_phase(
            data
        )

        if isinstance(
            result,
            JSONResponse
        ):

            return result

        return result

    return evaluate_phase(
        data
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "endpoint": "/bqml"
    }


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )