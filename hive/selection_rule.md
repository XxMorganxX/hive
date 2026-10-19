## How a proposed candidate is kept or discarded

Your proposed candidate (the child) is scored on the **search split** only, with the same
tasks, the same model and the same number of trials per task as the incumbent and the seed.

**Score.** Each task's reward is the mean over its trials. An errored trial counts as 0.
A trial is errored when no measurement was possible: a crash, a timeout, a provider or quota
failure, a harness that fails to load, or a grader failure. The score is the mean of these per-task rewards over every search task, so pushing a
hard task into an error never helps.

**Promotion rule.** The child replaces the incumbent only if both hold:

1. child score > incumbent score + $margin (the promotion margin), and
2. child score >= the seed's score (the starting candidate).

A tie, or a gain of $margin or less, keeps the incumbent.

**Held-out split.** A separate held-out split exists that you never see. It is scored once,
at the end of the experiment, on the seed and the final incumbent. Changes that only fit the
search tasks will not survive it, so aim for changes you expect to transfer.
