"""The MLflow inference protocol, embedded: parse and answer ``/invocations``.

Served models speak MLflow's REST scoring protocol exactly, so anything that can
call an ``mlflow models serve`` endpoint can call an elbi one: the request
is one of ``dataframe_split``, ``dataframe_records``, ``instances``, or ``inputs``
(with an optional ``params``), and the response is ``{"predictions": ...}``. Only
the payload handling lives here; the HTTP framing belongs to whichever server
embeds it (the app's FastAPI route, in practice).
"""

from __future__ import annotations

from typing import Any

from ..errors import ModelError

#: The mutually exclusive input framings the protocol accepts.
_FORMATS = ("dataframe_split", "dataframe_records", "instances", "inputs")


def parse_invocations(
    payload: Any, schema: Any = None
) -> tuple[Any, dict[str, Any] | None]:
    """The scoring input in ``payload`` as a pandas DataFrame, plus its ``params``.

    Exactly one of the protocol's input keys must be present, mirroring MLflow's own
    validation so error behavior matches the reference server. ``params`` passes
    through untouched (pyfunc validates it against the model's signature).

    ``schema`` is the model's input :class:`mlflow.types.Schema`; when given, the
    payload is parsed by MLflow's own scoring-server code, which casts JSON values
    to the schema's types exactly as ``mlflow models serve`` would (JSON has one
    number type, so an integer-looking ``1`` must become the double the model was
    trained on rather than failing strict enforcement).
    """
    import pandas as pd

    if not isinstance(payload, dict):
        raise ModelError(
            "the request body must be a JSON object with exactly one of: "
            + ", ".join(_FORMATS)
        )
    present = [key for key in _FORMATS if key in payload]
    if len(present) != 1:
        raise ModelError(
            "the request must specify exactly one of: "
            + ", ".join(_FORMATS)
            + (f" (got {', '.join(present)})" if present else "")
        )
    key = present[0]
    data = payload[key]
    params = payload.get("params")
    if params is not None and not isinstance(params, dict):
        raise ModelError("'params' must be a JSON object")
    try:
        if schema is not None:
            frame = _reference_parse({key: data}, schema)
        elif key == "dataframe_split":
            frame = pd.DataFrame(
                data=data.get("data"),
                columns=data.get("columns"),
                index=data.get("index"),
            )
        elif key == "dataframe_records":
            frame = pd.DataFrame.from_records(data)
        else:  # instances / inputs: records, a column map, or a bare array
            frame = _tensor_frame(pd, data)
    except ModelError:
        raise
    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelError(f"malformed {key!r} payload: {exc}") from exc
    # A tensor-schema parse can return an ndarray, which has size, not empty.
    empty = frame.empty if hasattr(frame, "empty") else getattr(frame, "size", 1) == 0
    if empty:
        raise ModelError(f"{key!r} contained no rows to score")
    return frame, params


def _reference_parse(data: dict[str, Any], schema: Any) -> Any:
    """Parse one framing with MLflow's scoring server, casting to ``schema``.

    This is the exact code path a standalone ``mlflow models serve`` runs, so type
    coercion, tensor handling, and error wording all match the reference server.
    Its errors surface as :class:`ModelError` (the protocol's 400), never a 500.
    """
    from mlflow.exceptions import MlflowException
    from mlflow.pyfunc import scoring_server

    try:
        return scoring_server.infer_and_parse_data(data, schema)
    except MlflowException as exc:
        raise ModelError(exc.message) from exc


def _tensor_frame(pd: Any, data: Any) -> Any:
    """A DataFrame for the TF-Serving style framings (``instances``/``inputs``)."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame.from_records(data)  # a list of named records
    # A column map ({"col": [...]}) or a bare array (columns are positional).
    return pd.DataFrame(data)


def predictions_payload(predictions: Any) -> dict[str, Any]:
    """``predictions`` as the protocol's JSON-safe response body."""
    if hasattr(predictions, "to_dict") and hasattr(predictions, "columns"):
        return {"predictions": predictions.to_dict(orient="records")}  # DataFrame
    if hasattr(predictions, "tolist"):
        return {"predictions": predictions.tolist()}  # ndarray / Series
    if isinstance(predictions, (list, tuple)):
        return {"predictions": [_json_safe(p) for p in predictions]}
    return {"predictions": _json_safe(predictions)}


def _json_safe(value: Any) -> Any:
    """A scalar as plain JSON (numpy scalars carry an ``item`` unwrapper)."""
    return value.item() if hasattr(value, "item") else value
