# `bishop`

Agentic AI research assistant

## NPCs implemented so far

* `Planner` (Chain of Thought) analyzes previous runs and formulates a hypothesis and plan for the next run
* `Implementer` (ReAct) codes up the plan for the next run, iteratively submitting code to a checker function and acting on feedback
* `Analyst` (ReAct) answers questions about a tabular dataset by calling a whitelisted subset of the `pandas` API
