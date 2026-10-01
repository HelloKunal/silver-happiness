"""Assembles the case workflow graph.

    intake -> triage --+--> refunds_specialist --+
                       +--> shipping_specialist -+--> (transfer? another specialist) --> policy_gate
                       +--> account_specialist --+                                          |
                       +--> general_specialist --+           +------------------------------+
                       +--> human_handoff                    |        |          |            |
                                                        execute  notify_approver respond  human_handoff
                                                            |        |
                                                            |   await_approval (interrupt: human decides via push)
                                                            |        |  approve -> execute | reject -> respond | takeover -> human_handoff
                                                            v
                                                         respond --> END            human_handoff --> END

Cross-cutting, for every node:
- RetryPolicy: transient infrastructure errors (DB locked, flaky dependency) retry with backoff.
- error_handler: anything still failing routes the case to human_handoff with the error attached.
"""

from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from support_agent.graph.nodes import CaseNodes
from support_agent.graph.state import CaseState
from support_agent.models import SPECIALISTS
from support_agent.resilience import TransientError
from support_agent.services import Services

SPECIALIST_NODES = [f"{name}_specialist" for name in SPECIALISTS]


def _retryable(exc: Exception) -> bool:
    # Only infrastructure hiccups. LLM failures were already retried + failed over by the gateway,
    # and re-running a node on a logic error just repeats it.
    return isinstance(exc, TransientError) or (isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc))


def _route(state: CaseState) -> str:
    return state["route"]


def build_graph(services: Services, checkpointer=None):
    n = CaseNodes(services)
    g = StateGraph(CaseState)
    g.set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5, retry_on=_retryable),
        error_handler=n.on_node_error,
    )

    g.add_node("intake", n.intake)
    g.add_node("triage", n.triage)
    for name in SPECIALISTS:
        g.add_node(f"{name}_specialist", n.specialist(name))
    g.add_node("policy_gate", n.policy_gate)
    g.add_node("notify_approver", n.notify_approver)
    g.add_node("await_approval", n.await_approval)
    g.add_node("execute_actions", n.execute_actions)
    g.add_node("respond", n.respond)
    g.add_node("human_handoff", n.human_handoff, error_handler=n.on_handoff_error)

    g.add_edge(START, "intake")
    g.add_edge("intake", "triage")
    g.add_conditional_edges("triage", _route, SPECIALIST_NODES + ["human_handoff"])
    for node in SPECIALIST_NODES:
        g.add_conditional_edges(node, _route, [s for s in SPECIALIST_NODES if s != node] + ["policy_gate"])
    g.add_conditional_edges("policy_gate", _route, ["execute_actions", "notify_approver", "respond", "human_handoff"])
    g.add_edge("notify_approver", "await_approval")
    g.add_conditional_edges("await_approval", _route, ["execute_actions", "respond", "human_handoff"])
    g.add_conditional_edges("execute_actions", _route, ["respond", "human_handoff"])
    g.add_edge("respond", END)
    g.add_edge("human_handoff", END)
    return g.compile(checkpointer=checkpointer)
