"""
Small Agent Pipeline Demo
==========================
A single-agent system built as a stateful directed graph (Q1), with:
  - Nodes & edges (Q2)          -> analyze -> route -> call_tool -> respond
  - Conditional routing (Q3)     -> picks calculator / keyword / general tool
  - A retry loop / cycle (Q4)    -> call_tool retries on transient failure
  - Multi-role single agent (Q5) -> analyzer, router, tool-caller, responder
  - JSON-schema-style tools (Q6) -> each tool declares input/output schema
  - Sequential tool calls (Q7)   -> steps run one after another (dependent)
  - Error handling (Q8)          -> try/except + retry + logging
  - Trajectory logging (Q9)      -> full step-by-step record kept in state
  - Completion rate & cost (Q10) -> tracked across a batch of queries

No external dependencies -- runs with plain Python 3.
"""

import re
import random
import statistics
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. TOOLS  (each has a simple JSON-schema-style input/output contract)
# ---------------------------------------------------------------------------

def calculator_tool(payload: dict) -> dict:
    """
    Input schema:  {"expression": str}
    Output schema: {"result": float}
    """
    expr = payload["expression"]
    if not re.fullmatch(r"[0-9\.\s\+\-\*/\(\)]+", expr):
        raise ValueError(f"Unsafe or invalid expression: {expr!r}")
    result = eval(expr)  # safe here: input is pre-validated to digits/operators only
    return {"result": result}


def keyword_tool(payload: dict) -> dict:
    """
    Input schema:  {"text": str}
    Output schema: {"keywords": list[str]}
    """
    stopwords = {"the", "a", "an", "is", "of", "in", "to", "for", "and", "on"}
    words = re.findall(r"[a-zA-Z]+", payload["text"].lower())
    keywords = sorted({w for w in words if w not in stopwords and len(w) > 2})
    return {"keywords": keywords}


def general_tool(payload: dict) -> dict:
    """
    Input schema:  {"text": str}
    Output schema: {"reply": str}
    """
    return {"reply": f"I heard: '{payload['text']}'. No specialized tool matched, so this is a general response."}


TOOL_REGISTRY = {
    "calculator": calculator_tool,
    "keyword": keyword_tool,
    "general": general_tool,
}


# ---------------------------------------------------------------------------
# 2. STATE  (what flows along the edges of the graph)
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    query: str
    route: str = None
    tool_input: dict = None
    tool_output: dict = None
    response: str = None
    trajectory: list = field(default_factory=list)   # Q9: full step record
    num_tool_calls: int = 0                            # Q10: cost metric
    num_retries: int = 0                                # Q10: cost metric
    success: bool = False                               # Q10: completion metric

    def log(self, node: str, detail: str):
        self.trajectory.append({"node": node, "detail": detail})


# ---------------------------------------------------------------------------
# 3. NODES  (Q2: each node is one task; edges are the function calls below)
# ---------------------------------------------------------------------------

def node_analyze(state: AgentState) -> AgentState:
    """Analyzer role (Q5): inspect the raw query, prep it for routing."""
    state.log("analyze", f"Received query: {state.query!r}")
    return state


def node_route(state: AgentState) -> AgentState:
    """Router role (Q5) / Conditional routing (Q3): pick a tool by simple rules."""
    q = state.query.lower()
    if "calculate" in q or re.search(r"\d+\s*[\+\-\*/]\s*\d+", q):
        state.route = "calculator"
        expr = re.sub(r"[^0-9\.\+\-\*/\(\)]", "", q)
        state.tool_input = {"expression": expr}
    elif "keywords" in q or "extract" in q:
        state.route = "keyword"
        state.tool_input = {"text": state.query}
    else:
        state.route = "general"
        state.tool_input = {"text": state.query}
    state.log("route", f"Routed to '{state.route}' tool")
    return state


def node_call_tool(state: AgentState, max_retries: int = 2, simulate_failures: int = 0) -> AgentState:
    """
    Tool-caller role (Q5). Sequential tool call (Q7) with a retry loop (Q4)
    and try/except error handling (Q8).
    """
    tool_fn = TOOL_REGISTRY[state.route]
    attempts_left = max_retries + 1
    failures_to_simulate = simulate_failures

    while attempts_left > 0:
        try:
            if failures_to_simulate > 0:
                failures_to_simulate -= 1
                raise ConnectionError("Simulated transient tool failure")
            state.tool_output = tool_fn(state.tool_input)
            state.num_tool_calls += 1
            state.log("call_tool", f"Tool '{state.route}' succeeded: {state.tool_output}")
            state.success = True
            return state
        except Exception as e:
            state.num_tool_calls += 1
            state.num_retries += 1
            attempts_left -= 1
            state.log("call_tool", f"Tool '{state.route}' failed ({e}); retries left: {attempts_left}")

    state.log("call_tool", f"Tool '{state.route}' failed after all retries")
    state.success = False
    return state


def node_respond(state: AgentState) -> AgentState:
    """Responder role (Q5): turn tool output into a final answer."""
    if not state.success:
        state.response = "Sorry, I couldn't complete that request after retrying."
    elif state.route == "calculator":
        state.response = f"Result: {state.tool_output['result']}"
    elif state.route == "keyword":
        state.response = f"Keywords: {', '.join(state.tool_output['keywords'])}"
    else:
        state.response = state.tool_output["reply"]
    state.log("respond", f"Final response: {state.response}")
    return state


# ---------------------------------------------------------------------------
# 4. GRAPH RUNNER  (Q1: stateful directed graph, edges chain the nodes)
# ---------------------------------------------------------------------------

def run_pipeline(query: str, simulate_failures: int = 0) -> AgentState:
    state = AgentState(query=query)
    state = node_analyze(state)
    state = node_route(state)
    state = node_call_tool(state, simulate_failures=simulate_failures)
    state = node_respond(state)
    return state


# ---------------------------------------------------------------------------
# 5. BATCH RUN + METRICS  (Q10: task completion rate & cost)
# ---------------------------------------------------------------------------

def run_batch(queries):
    results = []
    for q, fail_count in queries:
        results.append(run_pipeline(q, simulate_failures=fail_count))

    completion_rate = sum(r.success for r in results) / len(results)
    avg_tool_calls = statistics.mean(r.num_tool_calls for r in results)
    avg_retries = statistics.mean(r.num_retries for r in results)

    print("=" * 70)
    print("TRAJECTORIES")
    print("=" * 70)
    for r in results:
        print(f"\nQuery: {r.query}")
        for step in r.trajectory:
            print(f"   [{step['node']}] {step['detail']}")
        print(f"   -> Response: {r.response}")

    print("\n" + "=" * 70)
    print("BATCH METRICS")
    print("=" * 70)
    print(f"Task completion rate : {completion_rate:.0%}")
    print(f"Avg tool calls/query : {avg_tool_calls:.2f}")
    print(f"Avg retries/query    : {avg_retries:.2f}")
    return results


if __name__ == "__main__":
    demo_queries = [
        ("calculate 12 + 8 * 2", 0),          # routes to calculator, no failure
        ("extract keywords from this sentence about agent pipelines", 0),  # keyword tool
        ("tell me a fun fact", 0),             # falls through to general
        ("calculate 100 / 4", 1),              # calculator tool, 1 simulated failure -> retries
        ("calculate 5 * 5", 3),                # exceeds retry budget -> fails gracefully
    ]
    run_batch(demo_queries)
