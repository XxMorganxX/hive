# Framework adapters (Deep Agents, Strands) and extensions in the contract

## Goal

1. Explain the agent harness contract as it stands. Only Pi has a real adapter.
2. Make **extensions** a first-class, framework-neutral part of the contract. In Pi, extensions are TypeScript modules hooking the agent loop. They are a primary lever for harness evolution, but today they exist only inside the Pi adapter.
3. Add adapters for **LangChain Deep Agents** and **AWS Strands**, so the same candidate/suite/overseer loop runs on three frameworks.

## Approach

1. **Explore** (workflow, 4 parallel subagents):
   - explain the current contract;
   - scout a Deep Agents adapter against the installed package;
   - scout a Strands adapter against the installed package;
   - map Pi extensions, Deep Agents middleware and Strands hooks onto neutral lifecycle points.
2. **Design** how extensions fit the contract. Weigh minimal vs. rich designs; record the decision here and in `docs/FOUNDING_PRINCIPLES.md` §5.
3. **Implement:** I make the contract change myself, then fixers fan out on disjoint files (one adapter each, plus controller and Pi alignment). All work is test-first and offline, using fake models.
4. **Review** adversarially, then verify with real runs if credentials allow.

## Decisions

(recorded as they are made)

## Progress log

- 2026-10-19: Launched the exploration workflow.

## Open questions

- Credentials. Pi's only credential is an OpenAI key in `~/.pi/agent/auth.json`. Deep Agents (`langchain-openai`) and Strands (OpenAI provider) read `OPENAI_API_KEY` from the environment, so a real smoke run needs that variable set.

## Out of scope

- A portable extension API, where one extension file runs on every framework. Pending the mapping result.
