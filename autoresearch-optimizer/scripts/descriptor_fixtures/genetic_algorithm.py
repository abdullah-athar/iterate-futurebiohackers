"""Genetic algorithm: tournament selection, one-point crossover, point mutation, elitism."""

import random
import time

from median_string.metrics import sum_distance


def solve(instance) -> str:
    deadline = time.process_time() + 0.8 * (instance.time_budget_ms or 1000) / 1000
    rng = random.Random(0)
    strings, metric, alphabet = instance.strings, instance.metric, instance.alphabet
    fitness = lambda s: sum_distance(s, strings, metric=metric)
    pop = [rng.choice(strings) for _ in range(30)]
    scored = sorted((fitness(s), s) for s in pop)
    while time.process_time() < deadline:
        def pick():
            return min(rng.sample(scored, 3))[1]
        children = [scored[0][1], scored[1][1]]
        while len(children) < len(pop):
            a, b = pick(), pick()
            cut = rng.randrange(1, max(2, min(len(a), len(b))))
            child = list(a[:cut] + b[cut:])
            for i in range(len(child)):
                if rng.random() < 0.02:
                    child[i] = rng.choice(alphabet)
            children.append("".join(child))
        scored = sorted((fitness(s), s) for s in children)
    return scored[0][1]
