# AGENTS.md

## Project Context
You are a software engineer creating an agent for querying a Paperless NGX database, recommending existing tags for a set of documents or proposing new fitting tags where no existing tags are applicable. 
The results have to be reviewed by the user to be accepted or rejected.

## Code Strucutre

- the project uses LangGraph, so structure every behavior into nodes and corresponding graphs
- src contains all behavior for the agents 
- there are two different interaction strategies captured in src/strategies. (a) an iterative stategy where the classification and user interaction is performed for each document, i.e., first for the first document, then for the second and so on. (b) a sequential strategy where classifications are first performed for all documents, and then the review process is triggered. 
- src\core contains code common to both strategies. 
- test\paperless-instance-mock contains paperless content when no running paperless ngx instance is available. 


## Code Guidelines
- use behavor-specific names for methods, parameters and files. Avoid generic names like "payload" or similar. 
