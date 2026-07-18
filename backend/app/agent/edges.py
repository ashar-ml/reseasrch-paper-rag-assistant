from loguru import logger
from app.agent.state import AgentState

class AgentEdges:
    @staticmethod
    def decide_to_generate(state: AgentState) -> str:
        """
        Determines whether to route to the answer generation node or
        the query optimization/rewrite node based on chunk relevance.
        """
        documents = state["documents"]
        rewrite_count = state.get("rewrite_count", 0)
        
        logger.info(f"[LangGraph Edge: Router] Deciding routing path. Docs remaining: {len(documents)}, Rewrites: {rewrite_count}")
        
        # Limit total rewrites to 1 to prevent infinite query loops
        if rewrite_count >= 1:
            logger.info("  - Route target: 'generate' (Max query rewrites reached)")
            return "generate"
            
        # If there are active relevant chunks, proceed to generation
        if len(documents) > 0:
            logger.info("  - Route target: 'generate' (Relevant documents found)")
            return "generate"
            
        # Otherwise, rewrite the query to perform a better search
        logger.info("  - Route target: 'rewrite' (No relevant chunks graded)")
        return "rewrite"
