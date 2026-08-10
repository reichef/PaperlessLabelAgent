from paperlesslabelagent.core.state import AgentState


class SequentialAgentState(AgentState):
    """Extends the AgentState with the current iteration of the review-retry step over all documents"""

    iteration: int
