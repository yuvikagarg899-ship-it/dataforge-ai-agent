import sqlite3
import hashlib
import json
from datetime import datetime, timezone
import os 
from src.experiments.metrics import METRICS,canonical_metric

DB_PATH = os.environ.get(
    "EXPERIMENT_DB_PATH",
    "experiment_memory.db"
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_hash TEXT NOT NULL,
            task_type TEXT NOT NULL,
            target_column TEXT NOT NULL,
            model_type TEXT NOT NULL,
            features TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            baseline_metric_value REAL NOT NULL,
            beats_baseline INTEGER NOT NULL,
            accepted INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def hash_dataset(csv_path: str) -> str:
    with open(csv_path, "rb") as f:
        content = f.read()

    return hashlib.sha256(content).hexdigest()


def record_experiment(
    dataset_hash: str,
    objective,
    experiment_plan,
    results: dict,
    accepted: bool,
) -> None:

    conn = get_connection()

    conn.execute("""
        INSERT INTO experiments (
            dataset_hash,
            target_column,
            task_type,
            model_type,
            features,
            hyperparameters,
            metric_name,
            metric_value,
            baseline_metric_value,
            beats_baseline,
            accepted,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dataset_hash,
        objective.target_column,
        objective.task_type,
        experiment_plan.model_type,
        json.dumps(experiment_plan.features),
        json.dumps([
            {
                "name": hp.name,
                "value": hp.value
            }
            for hp in experiment_plan.hyperparameters
        ]),
        results["metric_name"],
        results["metric_value"],
        results["baseline_metric_value"],
        int(results["beats_baseline"]),
        int(accepted),
        datetime.now(timezone.utc).isoformat(),
    ))

    conn.commit()
    conn.close()


def get_prior_experiments(
    dataset_hash: str,
    target_column: str,
    task_type: str,
    top_n: int = 3,
    recent_n: int = 2,
    rejected_n: int = 3,
    max_combined:int = 8,
) -> dict:

    conn = get_connection()

    rows = conn.execute("""
        SELECT
            model_type,
            features,
            hyperparameters,
            metric_name,
            metric_value,
            baseline_metric_value,
            beats_baseline,
            accepted,
            created_at
        FROM experiments
        WHERE dataset_hash = ?
          AND target_column = ?
          AND task_type = ?
        ORDER BY created_at DESC
    """, (
        dataset_hash,
        target_column,
        task_type
    )).fetchall()

    conn.close()

    def row_to_dict(row):
        return {
            "model_type": row["model_type"],
            "features": json.loads(row["features"]),
            "hyperparameters": json.loads(row["hyperparameters"]),
            "metric_name": row["metric_name"],
            "metric_value": row["metric_value"],
            "baseline_metric_value": row["baseline_metric_value"],
            "beats_baseline": bool(row["beats_baseline"]),
            "accepted": bool(row["accepted"]),
            "created_at": row["created_at"],
        }

    experiments = [row_to_dict(r) for r in rows]

    def higher_is_better(metric_name: str) -> bool:
        key = canonical_metric(metric_name)
        return METRICS[key]["higher_is_better"] if key else True

    direction = higher_is_better(experiments[0]["metric_name"]) if experiments else True
    best = sorted(experiments, key=lambda e: e["metric_value"], reverse=direction)[:top_n]

    recent = sorted(
        experiments,
        key=lambda e: e["created_at"],
        reverse=True
    )[:recent_n]

    rejected = [
        e for e in experiments
        if not e["accepted"]
    ]

    rejected = sorted(
        rejected,
        key=lambda e: e["created_at"],
        reverse=True
    )[:rejected_n]

    def dedup(*lists):
        seen = set()
        result = []

        for lst in lists:
            for e in lst:
                key = (
                    e["model_type"],
                    tuple(sorted(e["features"])),
                    e["created_at"]
                )

                if key not in seen:
                    seen.add(key)
                    result.append(e)

        return result
    combined = dedup(best,recent,rejected)[:max_combined]

    return {
        "best": best,
        "recent": recent,
        "rejected": rejected,
        "combined": combined,
    }