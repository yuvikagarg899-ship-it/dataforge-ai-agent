from langchain_core.runnables import RunnableConfig
from src.profiling.profiler import profile_dataset
from src.planning.objective_parser import parse_objective
from src.graph.state import PipelineState
from src.cleaning.cleaning_planner import plan_cleaning
from src.cleaning.cleaning_executer import execute_cleaning_plan
from src.experiments.experiment_planner import plan_experiment
from src.experiments.experiment_executer import run_experiment
from src.reflection.reflector import reflect_on_Result
from src.memory.store import record_experiment,hash_dataset,get_prior_experiments
from src.critique.critic import critique_experiment
from src.graph.trace import trace_event
import pandas as pd
def profile_node(state: PipelineState, config: RunnableConfig) -> dict:
    profile = profile_dataset(state["csv_path"])
    return {
        "profile": profile,
        "trace": trace_event(
            state,
            "profile_node",
            f"{profile.n_rows} rows, {profile.n_columns} cols, flags={profile.dataset_level_flags}",
        ),
    }
def profile_for_df(profile, df):
    kept = [c for c in profile.columns if c.name in df.columns]
    return profile.model_copy(update={"columns": kept, "n_columns": len(kept)})

def parse_objective_node(
    state: PipelineState,
    config: RunnableConfig
) -> dict:
    client = config["configurable"]["client"]

    try:
        parsed = parse_objective(
            state["user_objective"],
            state["profile"],
            client
        )

        return {
            "parsed_objective": parsed,
            "error_message": None,
            "trace":trace_event(state,"parse_objective_node",f"target = {parsed.target_column},task = {parsed.task_type},metric = {parsed.target_metric}",),
        }

    except Exception as e:
        return {
           "error_message": str(e),
           "trace":trace_event(state,"parse_objective_node","FAILED to parsed objective"),
    }
def critique_experiment_node(state: PipelineState, config: RunnableConfig) -> dict:
    client = config["configurable"]["client"]
    profile = state["profile"]
    objective = state["parsed_objective"]
    plan = state["experiment_plan"]

    try:
        critique = critique_experiment(profile, objective, plan, client)
    except Exception as e:
        return {
            "error_message": f"Critique failed: {e}",
            "trace": trace_event(state, "critique_experiment_node", "FAILED to generate critique"),
        }

    current_count = state.get("critique_iteration_count", 0)
    return {
        "critique": critique,
        "critique_iteration_count": current_count + 1,
        "error_message": None,
        "trace": trace_event(
            state,
            "critique_experiment_node",
            f"verdict={critique.verdict}, concerns={len(critique.concerns)}, round={current_count + 1}",
        ),
    }
def cleaning_plan_node(state: PipelineState, config: RunnableConfig) -> dict:
    client = config["configurable"]["client"]
    profile = state["profile"]
    objective = state["parsed_objective"]

    try:
        plan = plan_cleaning(profile, objective, client)
        return {
            "cleaning_plan": plan,
            "error_message": None,
            "trace": trace_event(
                state, "cleaning_plan_node", f"{len(plan.actions)} cleaning actions proposed"
            ),
        }
    except Exception as e:
        return {
            "error_message": str(e),
            "trace": trace_event(state, "cleaning_plan_node", "FAILED to plan cleaning"),
        }



def execute_cleaning_node(state: PipelineState, config: RunnableConfig) -> dict:
    try:
        df = pd.read_csv(state["csv_path"])
        plan = state["cleaning_plan"]

        if plan.actions:
            df = execute_cleaning_plan(df, plan)

        return {
            "cleaned_df": df,
            "error_message": None,
            "trace": trace_event(
                state, "execute_cleaning_node", f"cleaned_df shape={df.shape}"
            ),
        }
    except Exception as e:
        return {
            "error_message": f"Cleaning failed: {e}",
            "trace": trace_event(state, "execute_cleaning_node", "FAILED to execute cleaning"),
        }

def plan_experiment_node(state: PipelineState, config: RunnableConfig) -> dict:
    client = config["configurable"]["client"]
    profile = profile_for_df(state["profile"], state["cleaned_df"])
    objective = state["parsed_objective"]

    dataset_hash = hash_dataset(state["csv_path"])
    memory = get_prior_experiments(
        dataset_hash=dataset_hash,
        target_column=objective.target_column,
        task_type=objective.task_type,
    )

    critique_feedback = None
    if state.get("critique") is not None and state["critique"].verdict == "reject":
        critique_feedback = state["critique"].concerns

    try:
        plan = plan_experiment(
            profile,
            objective,
            client,
            prior_experiments=memory["combined"],
            critique_feedback=critique_feedback,
        )
        return {
            "experiment_plan": plan,
            "error_message": None,
            "trace": trace_event(
                state,
                "plan_experiment_node",
                f"{plan.model_type}, {len(plan.features)} features, "
                f"memory_entries={len(memory['combined'])}, "
                f"critique_feedback={'yes' if critique_feedback else 'no'}",
            ),
        }
    except Exception as e:
        return {
            "error_message": str(e),
            "trace": trace_event(state, "plan_experiment_node", "FAILED to plan experiment"),
        }


def run_experiment_node(state: PipelineState, config: RunnableConfig) -> dict:
    df = state["cleaned_df"]
    plan = state["experiment_plan"]
    objective = state["parsed_objective"]

    try:
        results = run_experiment(df, plan, objective)
        return {
            "experiment_results": results,
            "error_message": None,
            "trace": trace_event(
                state,
                "run_experiment_node",
                f"{results['metric_name']}={results['metric_value']:.4f}, "
                f"baseline={results['baseline_metric_value']:.4f}, "
                f"beats_baseline={results['beats_baseline']}",
            ),
        }
    except Exception as e:
        return {
            "error_message": f"Training failed: {e}",
            "trace": trace_event(state, "run_experiment_node", "FAILED during training"),
        }


def reflect_node(state: PipelineState, config: RunnableConfig) -> dict:
    client = config["configurable"]["client"]
    objective = state["parsed_objective"]
    results = state["experiment_results"]
    target_col = next(
        (c for c in state["profile"].columns if c.name == objective.target_column),None
    )
    target_stats = {"mean":target_col.mean,"std":target_col.std} if target_col else None

    try:
        reflection = reflect_on_Result(objective, results, client,target_stats)
    except Exception as e:
        return {
            "error_message": f"Reflection failed: {e}",
            "trace": trace_event(state, "reflect_node", "FAILED to reflect"),
        }

    current_iteration = state.get("iteration_count", 0)
    accepted = reflection.decision == "accept"

    return {
        "reflection": reflection,
        "iteration_count": current_iteration + 1,
        "accepted": accepted,
        "critique": None,
        "critique_iteration_count": 0,
        "error_message": None,
        "trace": trace_event(
            state,
            "reflect_node",
            f"decision={reflection.decision}, experiment_iteration={current_iteration + 1}",
        ),
    }
def record_experiment_node(state: PipelineState, config: RunnableConfig) -> dict:
    dataset_hash = hash_dataset(state["csv_path"])
    record_experiment(
        dataset_hash=dataset_hash,
        objective=state["parsed_objective"],
        experiment_plan=state["experiment_plan"],
        results=state["experiment_results"],
        accepted=state["accepted"],
    )
    return {
        "trace": trace_event(
            state,
            "record_experiment_node",
            f"saved to memory (accepted={state['accepted']})",
        ),
    }