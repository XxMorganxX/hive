# Suite: python_functions

Each task asks the agent to implement one small, self-contained Python function.

## What a task looks like

- The **instruction** names the function and specifies its behaviour precisely: inputs, output type and edge cases.
- The agent's working directory starts with one file, `solution.py`. It holds the function's signature and docstring, and its body raises `NotImplementedError`.
- The agent must edit `solution.py` in place, keep the function's name and signature, use only the Python standard library, and must not print or read input.

## How grading works

- After the run, the grader copies `solution.py` into a fresh temporary directory and imports it in a separate Python process (isolated mode, with a time limit). The agent's final chat message is not graded; only `solution.py` is.
- The grader calls the function on a fixed set of hidden test cases. Each case passes when the return value, after a JSON round trip, equals the expected value. Tuples therefore compare equal to lists, and `True`/`False` must be real booleans, not `1`/`0`.
- A case fails as a **wrong answer** when the value differs or is not JSON-serializable, and fails with the **exception type name** when the call raises.
- If `solution.py` is missing, cannot be imported, or does not define the function, every case fails. If grading exceeds the time limit (for example, an infinite loop), every case fails.
- The grade detail reports how many hidden cases passed and counts the failures by kind. It never shows the hidden inputs or expected values.

## Metric

A task's reward is the fraction of its hidden cases that pass, in [0, 1]. The suite score is the mean reward over tasks.
