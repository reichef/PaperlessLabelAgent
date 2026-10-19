# AGENTS.md

## Project Context
You are a software engineer creating an agent for querying a Paperless NGX database, recommending existing tags for a set of documents or proposing new fitting tags where no existing tags are applicable. 
The results have to be reviewed by the user to be accepted or rejected.

## Code Strucutre

- the project uses LangGraph, so structure every behavior into nodes and corresponding graphs
- src contains all behavior for the agents 
- interaction strategies are captured in src/strategies. Currently there is only the iterative strategy, where the classification and user interaction is performed for each document, i.e., first for the first document, then for the second and so on. Each confirmed document is persisted immediately.
- a sequential strategy (classify all documents first, then review them) was removed because unpersisted work is lost if a run aborts, which limits it to very few documents. Its advantage is that it needs the user's attention only once. It is preserved on the branch `archive/sequential-strategy` for a possible later reintroduction.
- src\core contains code common to all strategies. 
- test\paperless-instance-mock contains paperless content when no running paperless ngx instance is available. 


## Code Guidelines
- use behavor-specific names for methods, parameters and files. Avoid generic names like "payload" or similar. 
