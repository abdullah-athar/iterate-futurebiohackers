# Team tasks

Claim a workstream by replacing `Unclaimed` with your name. Update the status and link the branch or PR so teammates can see what is in progress.

| Workstream | First deliverable | Owner | Status / branch or PR |
| --- | --- | --- | --- |
| Benchmark and baseline | Choose the problem, data split, metric direction, compute budget, and a reproducible baseline command. | Team | Done: `median_string` benchmark with Set Median baseline |
| Evaluation | Define candidate inputs and score outputs; report failures, elapsed time, and reproducibility metadata. | Team | Done: `median_string.Evaluator` with small/medium/hard tiers and JSON reports |
| Search loop | Propose candidates, evaluate under the agreed budget, and retain the best valid result. | Johann | In progress: `autoresearch/` greedy LLM loop, branch `johann/median-string-autoresearch` |
| Experiment analysis | Compare against the baseline and summarize improvements, failure cases, and limitations. | Unclaimed | To do |
| Demo and submission | Prepare the live demo, short description, credits, and presentation. | Unclaimed | To do |

## Decisions to agree first

- [x] Select the team's official track and benchmark: Track 1 (Autoresearch) on Median String (Steiner String) problem.
- [x] Specify the objective, success threshold, and evaluation budget: Minimize total Steiner distance vs. Set Median baseline.
- [x] Agree on the candidate interface and result format: `BaseSolver.solve(instance)` returning candidate string, evaluated by `median_string.Evaluator`.

- [x] Decide how to separate search-time feedback from final evaluation: search on the canonical tier, report the champion on the same tier regenerated with seeds offset by 10,000.
- [ ] Choose the compute provider and assign someone to save all work locally before temporary access expires.
