from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.agent.nodes import AgentNodes
from app.agent.edges import AgentEdges

# 1. Initialize State Graph Builder
workflow = StateGraph(AgentState)

# 2. Register Nodes
workflow.add_node("retrieve", AgentNodes.retrieve_node)
workflow.add_node("grade_documents", AgentNodes.grade_documents_node)
workflow.add_node("rewrite_query", AgentNodes.rewrite_query_node)
workflow.add_node("generate", AgentNodes.generate_node)

# 3. Establish Connections and Flow Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Add Conditional Route from Document Grading
workflow.add_conditional_edges(
    "grade_documents",
    AgentEdges.decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite_query"
    }
)

# Connect Rewrite Node back to Retrieval for iterative Corrective check
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

# 4. Compile Flow
app_graph = workflow.compile()
