# train_xor.py
import random

from src.core.innovation import InnovationTracker
from src.core.genome import Genome, NodeGene, ConnectionGene
from src.evolve.speciation import Speciator
from src.evolve.mutation import mutate_weights, mutate_add_connection, mutate_add_node
from src.evolve.crossover import crossover
from src.evolve.loop import run_training_loop

def make_initial_population_factory(pop_size, tracker):
    def _make():
        pop = []
        for _ in range(pop_size):
            g = Genome()
            # nœuds initiaux
            g.add_node(NodeGene(1, "input"))
            g.add_node(NodeGene(2, "input"))
            g.add_node(NodeGene(3, "bias"))
            g.add_node(NodeGene(4, "output"))
            # Connexions directes vers la sortie (option pédagogique)
            inn_14 = tracker.get_or_create(1, 4)
            inn_24 = tracker.get_or_create(2, 4)
            inn_34 = tracker.get_or_create(3, 4)
            # poids aléatoires
            g.add_connection(ConnectionGene(1, 4, random.uniform(-1,1), True, inn_14))
            g.add_connection(ConnectionGene(2, 4, random.uniform(-1,1), True, inn_24))
            g.add_connection(ConnectionGene(3, 4, random.uniform(-1,1), True, inn_34))
            pop.append(g)
        return pop
    return _make

if __name__ == "__main__":
    random.seed(123)

    cfg = {
        "population": 150,
        "elitism_per_species": 1,
        "mutation": {
            "p_mutate_weight": 0.8,
            "p_reset_weight": 0.1,
            "weight_sigma": 0.5,
            "p_add_conn": 0.2,
            "p_add_node": 0.2,
        }
    }

    tracker = InnovationTracker()
    speciator = Speciator(
        c1=1.0, c2=1.0, c3=0.4,
        delta_threshold=0.5,
        target_species=6,
        adjust_every=10,
        adjust_step=0.2
    )

    population, history = run_training_loop(
        make_initial_population=make_initial_population_factory(cfg["population"], tracker),
        tracker=tracker,
        speciator=speciator,
        cfg=cfg,
        max_generations=1000,
        success_mse=0.01,
        crossover_fn=crossover,
        mutate_weight_fn=mutate_weights,
        mutate_add_conn_fn=mutate_add_connection,
        mutate_add_node_fn=mutate_add_node,
        verbose=True
    )
