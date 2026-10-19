"""The model gateway: the controller's single door to model providers (P3, P12).

See tasks/framework-adapters-and-extensions.md, "Model gateway (spec)". Public API:

    with ModelGateway(model=..., prices=..., upstream_url=..., upstream_key=...,
                      max_calls_per_trial=...) as gateway:
        trial_key, endpoint = gateway.open_trial()
        ...agent.run(task, workdir, trace_dir, endpoint)...
        ledger = gateway.close_trial(trial_key)
"""

from hive.gateway.gateway import GatewayCall, ModelGateway, TrialLedger

__all__ = ["GatewayCall", "ModelGateway", "TrialLedger"]
