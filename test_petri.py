import pytest
from petri import PetriNet


def make():
    return (PetriNet(seed=0).add_place("a", 2).add_place("b")
            .add_transition("t", {"a": 1}, {"b": 1}))


def test_fire_moves_tokens():
    n = make()
    n.fire("t")
    assert n.marking == {"a": 1, "b": 1}


def test_disabled_raises():
    n = make()
    n.fire("t"); n.fire("t")
    with pytest.raises(ValueError):
        n.fire("t")


def test_run_stops_at_deadlock():
    assert make().run() == ["t", "t"]


def test_reachable_counts():
    _, states, _ = make().reachable()
    assert len(states) == 3
