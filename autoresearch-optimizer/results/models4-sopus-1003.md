# Sopus: Sonnet for generation 1, then Opus (4 agents × 2 generations, mid objective, 2026-10-03)

Idea: Sonnet already reaches a good first minimum cheaply in generation 1, so let it open the search
and hand generation 2 to Opus to push the minimum further. Settings are those of the rerun in
[models4-rerun-1003.md](models4-rerun-1003.md), with `--model sonnet,opus` (a per-generation schedule:
generation 1 on Sonnet, every later one on Opus). Dashboard, with the Sopus run next to the three
single-model reruns and buttons to show or hide each run:
[models4-sopus-dashboard-1003.html](models4-sopus-dashboard-1003.html).

| | Haiku | Sonnet | Opus | Sopus (Sonnet → Opus) |
|---|---|---|---|---|
| Best on validate (seed 22,816; planted reference 18,787) | 20,415 | 19,172 | **18,718** | 18,846 |
| Gain vs seed | 10.5% | 16.0% | **18.0%** | 17.4% |
| Holdout best (seed 22,575) | 20,393 | 18,920 | **18,628** | 18,629 |
| Agent tokens | 4.05 M | **0.67 M** | 0.96 M | 0.76 M |
| Agent cost (Claude Code estimate) | $1.17 | **$0.54** | $1.34 | $0.89 |
| Objective points gained per dollar | 2,050 | **6,716** | 3,049 | 4,478 |
| Wall clock (ledger, first to last entry) | 5.4 min | **1.5 min** | 2.4 min | 2.0 min |

- Generation 1 (Sonnet, 4 sessions, $0.30) reached 19,136, already better than Sonnet's final best in its
  own run. Generation 2 (Opus, $0.59) brought it to 18,846.
- Sopus matches Opus on the held-out instances (18,629 vs 18,628) for **34% less** ($0.89 vs $1.34), and
  is 0.6 points of gain behind Opus on validate. It sits between Sonnet and Opus on points per dollar.
- It stays just above the planted reference on validate (18,846 vs 18,787), where Opus went below it.
- One run per configuration: differences of under ~1% are within noise. A second Sopus run, or more
  generations (`--model sonnet,opus` with `--generations 3`), would tell whether the gap to Opus closes.
