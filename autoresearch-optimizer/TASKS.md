# Team tasks

Claim a workstream by replacing `Unclaimed` with your name. Update the status and link the branch or PR so teammates can see what is in progress.

| Workstream | First deliverable | Owner | Status / branch or PR |
| --- | --- | --- | --- |
| Benchmark and baseline | Choose the problem, data split, metric direction, compute budget, and a reproducible baseline command. | Team | Done: `median_string` benchmark with Set Median baseline |
| Evaluation | Define candidate inputs and score outputs; report failures, elapsed time, and reproducibility metadata. | Team | Done: `median_string.Evaluator` with small/medium/hard tiers and JSON reports |
| Search loop | Propose candidates, evaluate under the agreed budget, and retain the best valid result. | Abdullah + Devin | Done: `autoresearch/` (novelty gate, cascade, confirm re-test, Pareto archive, mode bandit; API + agent mode) |
| Experiment analysis | Compare against the baseline and summarize improvements, failure cases, and limitations. | Abdullah + Devin | In progress: `python -m autoresearch report --holdout` renders trajectory, front, verdicts, tokens, held-out |
| Demo and submission | Prepare the live demo, short description, credits, and presentation. | Unclaimed | To do |

## Decisions to agree first

- [x] Select the team's official track and benchmark: Track 1 (Autoresearch) on Median String (Steiner String) problem.
- [x] Specify the objective, success threshold, and evaluation budget: Minimize total Steiner distance vs. Set Median baseline.
- [x] Agree on the candidate interface and result format: `BaseSolver.solve(instance)` returning candidate string, evaluated by `median_string.Evaluator`.

- [x] Separate search-time feedback from final evaluation: search sees `small`/`medium`; claimed bests are re-tested on fresh `confirm` seeds; the report uses an unseen `holdout` suite.
- [ ] Choose the compute provider and assign someone to save all work locally before temporary access expires.
