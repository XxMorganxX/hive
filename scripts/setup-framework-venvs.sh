#!/usr/bin/env bash
# Create one pinned virtualenv per Python agent framework under .venvs/<framework>/.
#
# Why separate venvs: HIVE's controller has no dependencies, and each framework pulls in
# 50+ packages that pin conflicting versions (Deep Agents needs openai 3.x, Strands needs
# openai 2.x). Each adapter runs its trials with its own venv's interpreter, named in the
# experiment config as [agent] python_path.
#
# Usage: scripts/setup-framework-venvs.sh [deepagents] [strands]   (default: all)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frameworks=("$@")
if [ ${#frameworks[@]} -eq 0 ]; then
  frameworks=(deepagents strands)
fi

for framework in "${frameworks[@]}"; do
  requirements="$repo_root/hive/agents/$framework/requirements.txt"
  venv="$repo_root/.venvs/$framework"
  if [ ! -f "$requirements" ]; then
    echo "no pinned requirements for '$framework' at $requirements" >&2
    exit 1
  fi
  echo "==> $framework: $venv"
  uv venv --quiet --allow-existing --python 3.12 "$venv"
  uv pip install --quiet --python "$venv/bin/python" -r "$requirements"
done
