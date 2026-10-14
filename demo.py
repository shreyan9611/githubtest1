"""Producer/consumer with a bounded buffer of size 3."""
from petri import PetriNet

net = (
    PetriNet(seed=1)
    .add_place("idle_producer", 1)
    .add_place("buffer", 0)
    .add_place("free_slots", 3)
    .add_place("idle_consumer", 1)
    .add_transition("produce", {"idle_producer": 1, "free_slots": 1}, {"idle_producer": 1, "buffer": 1})
    .add_transition("consume", {"idle_consumer": 1, "buffer": 1}, {"idle_consumer": 1, "free_slots": 1})
)

net.run(max_steps=10, verbose=True)
places, states, edges = net.reachable()
print(f"\nreachable markings: {len(states)} (places: {places})")
