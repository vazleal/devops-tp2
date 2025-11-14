from __future__ import annotations

import os
import pickle
import threading
from collections import defaultdict
from typing import Iterable, List

from flask import Flask, jsonify, request

MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/model/model.pkl")
DEFAULT_LIMIT = int(os.getenv("MAX_RECOMMENDATIONS", "20"))
VERSION = os.getenv("VERSION", "1.0.0")

app = Flask(__name__)
app.model = None
app.model_mtime = 0.0
app._model_lock = threading.Lock()


class ModelNotReady(RuntimeError):
    """Raised when the model could not be loaded from disk."""


def _normalise_track(name: str) -> str:
    return name.strip().lower()


def _ensure_iterable(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _prepare_rule(raw_rule: dict | list | tuple, index: int) -> dict | None:
    """Normalise a rule entry stored in the pickled model."""

    antecedent = raw_rule.get("antecedent") if isinstance(raw_rule, dict) else None
    consequent = raw_rule.get("consequent") if isinstance(raw_rule, dict) else None
    support = raw_rule.get("support") if isinstance(raw_rule, dict) else None
    confidence = raw_rule.get("confidence") if isinstance(raw_rule, dict) else None

    if antecedent is None or consequent is None:
        if isinstance(raw_rule, (list, tuple)) and len(raw_rule) >= 2:
            antecedent = raw_rule[0]
            consequent = raw_rule[1]
            numeric_tail = [value for value in raw_rule[2:] if isinstance(value, (int, float))]
            if numeric_tail:
                confidence = float(numeric_tail[-1])
                if len(numeric_tail) > 1:
                    support = float(numeric_tail[0])
        else:
            return None

    antecedent_list = [item for item in _ensure_iterable(antecedent) if isinstance(item, str) and item.strip()]
    consequent_list = [item for item in _ensure_iterable(consequent) if isinstance(item, str) and item.strip()]

    if not antecedent_list or not consequent_list:
        return None

    return {
        "antecedent": antecedent_list,
        "consequent": consequent_list,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "support": float(support) if support is not None else None,
        "index": index,
    }


def _load_rules(model: dict) -> List[dict]:
    rules = []
    for index, raw_rule in enumerate(model.get("rules", [])):
        parsed = _prepare_rule(raw_rule, index)
        if parsed:
            antecedent_norm = {_normalise_track(track) for track in parsed["antecedent"]}
            consequent_norm = [_normalise_track(track) for track in parsed["consequent"]]
            parsed["antecedent_norm"] = antecedent_norm
            parsed["consequent_norm"] = consequent_norm
            rules.append(parsed)
    return rules


def load_model(force: bool = False) -> dict:
    """Load the pickled model from disk, reloading if it changed."""

    try:
        mtime = os.path.getmtime(MODEL_PATH)
    except OSError as exc:  # pragma: no cover - runtime guard
        raise ModelNotReady(str(exc)) from exc

    with app._model_lock:
        if force or mtime != app.model_mtime:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)
            model["_compiled_rules"] = _load_rules(model)
            app.model = model
            app.model_mtime = mtime
    if app.model is None:
        raise ModelNotReady("Model could not be loaded")
    return app.model


def compute_recommendation(model: dict, songs: Iterable[str], limit: int | None = None) -> list[dict]:
    """Compute recommendations using the association rules model."""

    limit = limit or DEFAULT_LIMIT
    if limit <= 0:
        return []

    user_lookup = {}
    for song in songs or []:
        if not isinstance(song, str):
            continue
        cleaned = song.strip()
        if not cleaned:
            continue
        key = _normalise_track(cleaned)
        user_lookup.setdefault(key, cleaned)

    if not user_lookup:
        return []

    compiled_rules = model.get("_compiled_rules") or []
    candidate_scores = defaultdict(float)
    candidate_details = defaultdict(list)
    display_values = {}

    for rule in compiled_rules:
        antecedent_norm = rule.get("antecedent_norm", set())
        if not antecedent_norm:
            continue
        if not antecedent_norm.issubset(user_lookup.keys()):
            continue

        support = rule.get("support") or 0.0
        confidence = rule.get("confidence") or 0.0
        weight = confidence
        if support:
            weight *= 0.7 + 0.3 * support
        weight *= 1.0 + 0.05 * len(rule["antecedent"])

        for original, normalised in zip(rule["consequent"], rule["consequent_norm"]):
            if normalised in user_lookup:
                continue
            candidate_scores[normalised] += weight
            display_values.setdefault(normalised, original)
            candidate_details[normalised].append(
                {
                    "from_rule": rule["index"],
                    "confidence": confidence,
                    "support": support or None,
                    "antecedent": rule["antecedent"],
                }
            )

    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: (item[1], max((detail["confidence"] for detail in candidate_details[item[0]]), default=0.0)),
        reverse=True,
    )

    recommendations = []
    for normalised, score in ranked[:limit]:
        details = candidate_details[normalised]
        top_conf = max((entry["confidence"] for entry in details), default=0.0)
        avg_conf = sum(entry["confidence"] for entry in details) / len(details) if details else 0.0
        recommendations.append(
            {
                "track": display_values.get(normalised, normalised),
                "score": round(score, 6),
                "max_confidence": round(top_conf, 6),
                "avg_confidence": round(avg_conf, 6),
                "support": max((entry.get("support") or 0.0 for entry in details), default=None),
                "evidence": details,
            }
        )

    return recommendations


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        model = load_model()
    except ModelNotReady:
        return jsonify({"status": "degraded", "model": False}), 503
    return jsonify({"status": "ok", "model": True, "rules": len(model.get("_compiled_rules", []))})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    try:
        model = load_model()
    except ModelNotReady as exc:
        return jsonify({"error": str(exc)}), 503

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "A JSON body with a 'songs' field is required"}), 400

    songs = payload.get("songs")
    if songs is None:
        return jsonify({"error": "Missing 'songs' in request body"}), 400

    limit = payload.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return jsonify({"error": "'limit' must be an integer"}), 400

    recommendations = compute_recommendation(model, songs, limit=limit)

    return jsonify(
        {
            "songs": [item["track"] for item in recommendations],
            "details": recommendations,
            "version": VERSION,
            "model_date": model.get("created_at", app.model_mtime),
            "input_count": len(songs) if isinstance(songs, list) else None,
        }
    )

@app.route("/api/model", methods=["GET"])
def model_metadata():
    try:
        model = load_model()
    except ModelNotReady as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(
        {
            "version": VERSION,
            "created_at": model.get("created_at"),
            "rules": len(model.get("_compiled_rules", [])),
            "params": model.get("params", {}),
        }
    )

if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "50023")))