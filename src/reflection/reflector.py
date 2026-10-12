from google import genai
from src.reflection.schema import ReflectionResult
from src.experiments.metrics import METRICS, canonical_metric

# The dummy baseline scores near zero on these by construction, so beating it says nothing
UNINFORMATIVE_BASELINE = {"recall", "precision", "f1", "r2"}


def reflect_on_Result(objective, experiment_results, client, target_stats=None):
    if not experiment_results["beats_baseline"]:
        return ReflectionResult(
            decision="retry",
            reasoning=(
                f"The model's {experiment_results['metric_name']} "
                f"({experiment_results['metric_value']:.4f}) did not beat the baseline "
                f"({experiment_results['baseline_metric_value']:.4f}). Retrying is required."
            ),
        )

    metric_key = canonical_metric(experiment_results["metric_name"])
    spec = METRICS.get(metric_key)
    value = experiment_results["metric_value"]
    baseline = experiment_results["baseline_metric_value"]

    lines = [f"- {metric_key}: {value:.4f}", f"- baseline {metric_key}: {baseline:.4f}"]

    if spec and not spec["higher_is_better"]:
        lines.append("- this is an error metric: LOWER is better")
        if baseline > 0:
            lines.append(f"- the model's error is {value / baseline:.0%} of the baseline's error")
        if target_stats and target_stats.get("mean"):
            lines.append(
                f"- target mean: {target_stats['mean']:.2f}, std: {target_stats['std']:.2f}"
            )
            lines.append(f"- the error is {value / target_stats['mean']:.1%} of the target mean")

    result_block = "\n".join(lines)

    if metric_key in UNINFORMATIVE_BASELINE:
        baseline_note = (
            "The baseline scores near zero on this metric by construction, so beating it is "
            "NOT evidence of quality. Do not cite it. Judge the absolute value."
        )
    else:
        baseline_note = (
            "The baseline is a naive model (it always predicts the most common class or the "
            "mean), so the gap to it is meaningful context."
        )

    prompt = f"""You are reviewing the result of a machine learning experiment.

User's objective:
- task_type: {objective.task_type}
- target_metric: {objective.target_metric}
- reasoning at parse time: {objective.reasoning}
- stated constraints: {objective.constraints}

Experiment result ({experiment_results['model_type']}):
{result_block}

{baseline_note}

Decide whether this result genuinely satisfies the user's stated objective, judging
against the numbers above, or whether another experiment should be attempted. If you
decide to retry, explain concretely what should change (a different model, different
hyperparameters, different features) and why, not just that the score could theoretically
be higher.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ReflectionResult,
        },
    )
    return ReflectionResult.model_validate_json(response.text)