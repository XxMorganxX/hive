"""The LangChain Deep Agents adapter. See `adapter.py` (controller side), `runner.py` (runs in the
framework's own venv) and `VALVES.md`."""

from hive.agents.deepagents.adapter import DeepAgentsAdapter

__all__ = ["DeepAgentsAdapter"]
