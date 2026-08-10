from paperlesslabelagent.core.state import AgentState, FileProposal


class IterativeFileProposal(FileProposal):
    """Extends FileProposal with an for the iterative strategy by an per-document retry counter."""

    attempts: int


class IterativeAgentState(AgentState):
    """Extends AgentState with an marker about which is the current file targeted in the iteration."""

    current_filename: str | None
