# `bishop`

Agentic AI research assistant

## NPCs implemented so far

* `Ideator` (Chain of Thought) generates hypotheses based on previous results
* `Planner` (Chain of Thought) selects one hypothesis and formulates a more detailed plan
* `Coder` (ReAct) codes up the plan for the next run, iteratively submitting code to a checker function and acting on feedback
* `Analyst` (ReAct) answers questions about a tabular dataset by calling a whitelisted subset of the `pandas` API
