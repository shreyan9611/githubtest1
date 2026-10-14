"""A small place/transition Petri net simulator with weighted arcs."""
import random
from dataclasses import dataclass, field


@dataclass
class Transition:
    name: str
    inputs: dict = field(default_factory=dict)   # place -> weight consumed
    outputs: dict = field(default_factory=dict)  # place -> weight produced


class PetriNet:
    def __init__(self, seed=None):
        self.marking = {}
        self.transitions = {}
        self.rng = random.Random(seed)

    def add_place(self, name, tokens=0):
        self.marking[name] = tokens
        return self

    def add_transition(self, name, inputs=None, outputs=None):
        inputs, outputs = dict(inputs or {}), dict(outputs or {})
        for p in (*inputs, *outputs):
            if p not in self.marking:
                raise KeyError(f"unknown place {p!r}")
        self.transitions[name] = Transition(name, inputs, outputs)
        return self

    def is_enabled(self, name):
        t = self.transitions[name]
        return all(self.marking[p] >= w for p, w in t.inputs.items())

    def enabled(self):
        return [n for n in self.transitions if self.is_enabled(n)]

    def fire(self, name):
        if not self.is_enabled(name):
            raise ValueError(f"transition {name!r} is not enabled")
        t = self.transitions[name]
        for p, w in t.inputs.items():
            self.marking[p] -= w
        for p, w in t.outputs.items():
            self.marking[p] += w

    def step(self):
        """Fire one randomly chosen enabled transition. Returns its name, or None if dead."""
        choices = self.enabled()
        if not choices:
            return None
        name = self.rng.choice(choices)
        self.fire(name)
        return name

    def run(self, max_steps=100, verbose=False):
        """Run until deadlock or max_steps. Returns the list of fired transitions."""
        trace = []
        if verbose:
            print(f"start: {self.marking}")
        for _ in range(max_steps):
            name = self.step()
            if name is None:
                if verbose:
                    print("deadlock")
                break
            trace.append(name)
            if verbose:
                print(f"{name:>10} -> {self.marking}")
        return trace

    def reachable(self, limit=10000):
        """Breadth-first reachability graph. Returns (markings, edges); markings are tuples."""
        places = sorted(self.marking)
        start = tuple(self.marking[p] for p in places)
        seen, edges, queue = {start}, [], [start]
        saved = dict(self.marking)
        try:
            while queue:
                m = queue.pop(0)
                for t in self.transitions:
                    self.marking = dict(zip(places, m))
                    if self.is_enabled(t):
                        self.fire(t)
                        n = tuple(self.marking[p] for p in places)
                        edges.append((m, t, n))
                        if n not in seen:
                            if len(seen) >= limit:
                                raise RuntimeError("state space exceeds limit (unbounded net?)")
                            seen.add(n)
                            queue.append(n)
        finally:
            self.marking = saved
        return places, seen, edges
