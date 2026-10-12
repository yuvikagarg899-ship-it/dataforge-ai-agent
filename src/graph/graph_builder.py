from langgraph.graph import StateGraph, END

from src.graph.state import PipelineState
from src.graph.nodes import (
    profile_node,
    parse_objective_node,
    reflect_node,
    cleaning_plan_node,
    execute_cleaning_node,
    plan_experiment_node,
    run_experiment_node,
    record_experiment_node,
    critique_experiment_node,
)


MAX_ITERATIONS = 3
CRITIQUE_MAX_ITERATIONS = 2


def should_retry(state: PipelineState) -> str:

    if state.get("error_message"):
        return END

    if state["accepted"]:
        return END

    if state["iteration_count"] >= MAX_ITERATIONS:
        return END

    return "plan_experiment_node"


def make_error_checker(next_node: str):

    def check(state: PipelineState) -> str:

        if state.get("error_message"):
            return END

        return next_node

    return check


def should_replan_after_critique(state: PipelineState) -> str:

    if state.get("error_message"):
        return END

    critique = state["critique"]
    if critique.verdict == "approve":
        return "run_experiment_node"
    if state["critique_iteration_count"] < CRITIQUE_MAX_ITERATIONS:
        return "plan_experiment_node"

    return "run_experiment_node"


def build_graph():

    graph = StateGraph(PipelineState)


    graph.add_node("profile_node", profile_node)

    graph.add_node(
        "parse_objective_node",
        parse_objective_node
    )

    graph.add_node(
        "cleaning_plan_node",
        cleaning_plan_node
    )

    graph.add_node(
        "execute_cleaning_node",
        execute_cleaning_node
    )

    graph.add_node(
        "plan_experiment_node",
        plan_experiment_node
    )

    graph.add_node(
        "critique_experiment_node",
        critique_experiment_node
    )

    graph.add_node(
        "run_experiment_node",
        run_experiment_node
    )

    graph.add_node(
        "reflect_node",
        reflect_node
    )

    graph.add_node(
        "record_experiment_node",
        record_experiment_node
    )


    graph.set_entry_point("profile_node")

    graph.add_edge(
        "profile_node",
        "parse_objective_node"
    )


    graph.add_conditional_edges(
        "parse_objective_node",
        make_error_checker("cleaning_plan_node"),
        {
            END: END,
            "cleaning_plan_node": "cleaning_plan_node",
        },
    )

    graph.add_conditional_edges(
        "cleaning_plan_node",
        make_error_checker("execute_cleaning_node"),
        {
            END: END,
            "execute_cleaning_node": "execute_cleaning_node",
        },
    )

    graph.add_conditional_edges(
        "execute_cleaning_node",
        make_error_checker("plan_experiment_node"),
        {
            END: END,
            "plan_experiment_node": "plan_experiment_node",
        },
    )


    graph.add_conditional_edges(
        "plan_experiment_node",
        make_error_checker("critique_experiment_node"),
        {
            END: END,
            "critique_experiment_node": "critique_experiment_node",
        },
    )

    graph.add_conditional_edges(
        "critique_experiment_node",
        should_replan_after_critique,
        {
            END: END,
            "plan_experiment_node": "plan_experiment_node",
            "run_experiment_node": "run_experiment_node",
        },
    )


    graph.add_conditional_edges(
        "run_experiment_node",
        make_error_checker("reflect_node"),
        {
            END: END,
            "reflect_node": "reflect_node",
        },
    )
    graph.add_conditional_edges(
        "reflect_node",
        make_error_checker("record_experiment_node"),
        {END:END,"record_experiment_node":"record_experiment_node"},
    )

    graph.add_conditional_edges(
        "record_experiment_node",
        should_retry,
        {
            END: END,
            "plan_experiment_node": "plan_experiment_node",
        },
    )

    return graph.compile()