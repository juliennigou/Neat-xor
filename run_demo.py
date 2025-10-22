# run_demo.py
import random
import math

# --- Imports de ton projet ---
from src.core.innovation import InnovationTracker
from src.core.genome import NodeGene, ConnectionGene, Genome
from src.eval.xor_task import mse, fitness_from_mse, accuracy_threshold
from src.evolve.mutation import mutate_weights, mutate_add_connection, mutate_add_node
from src.evolve.crossover import crossover
from src.evolve.speciation import Speciator, compute_compatibility_distance

# ------------------------------
# Helpers d’affichage
# ------------------------------
def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_genome(g: Genome, title="GENOME"):
    print(f"\n[{title}]")
    nodes = sorted([(nid, n.type) for nid, n in g.nodes.items()])
    print("Nodes:", nodes)
    conns = []
    for (u, v), c in g.conns.items():
        conns.append((c.innovation, u, v, round(c.weight, 3), c.enabled))
    for row in sorted(conns, key=lambda t: t[0]):
        inn, u, v, w, en = row
        print(f"  inn={inn:>2}  {u}->{v}  w={w:>6}  enabled={en}")

# ------------------------------
# Config minimale mutations
# ------------------------------
CFG = {
    "mutation": {
        "p_reset_weight": 0.1,
        "weight_sigma": 0.5,
    }
}

# ------------------------------
# 1) InnovationTracker
# ------------------------------
def test_innovation_tracker():
    banner("1) InnovationTracker")
    tracker = InnovationTracker()
    a = tracker.get_or_create(1, 4)
    b = tracker.get_or_create(2, 4)
    c = tracker.get_or_create(1, 4)  # doit renvoyer a
    assert a == c, "Le même (in,out) doit rendre le même innovation id"
    print("OK: (1,4)->", a, "  (2,4)->", b, "  count=", tracker.count())
    return tracker

# ------------------------------
# 2) Construire un génome XOR simple
# ------------------------------
def build_simple_genome(tracker: InnovationTracker) -> Genome:
    g = Genome()
    # nœuds
    g.add_node(NodeGene(1, "input"))
    g.add_node(NodeGene(2, "input"))
    g.add_node(NodeGene(3, "bias"))
    g.add_node(NodeGene(4, "output"))
    # connexions avec innovations stables
    inn_14 = tracker.get_or_create(1, 4)
    inn_24 = tracker.get_or_create(2, 4)
    inn_34 = tracker.get_or_create(3, 4)

    g.add_connection(ConnectionGene(1, 4, 0.75, True, inn_14))
    g.add_connection(ConnectionGene(2, 4, -0.10, True, inn_24))
    g.add_connection(ConnectionGene(3, 4, 0.50, True, inn_34))
    return g

def test_forward_and_eval(g: Genome):
    banner("2) Forward + Évaluation XOR")
    # Quelques prédictions brutes
    print("y_hat(1,0) =", round(g.forward([1.0, 0.0]), 4))
    print("y_hat(0,1) =", round(g.forward([0.0, 1.0]), 4))
    print("y_hat(0,0) =", round(g.forward([0.0, 0.0]), 4))
    # MSE / Fitness / Accuracy
    m = mse(g)
    f = fitness_from_mse(m)
    acc = accuracy_threshold(g, threshold=0.5)
    print(f"MSE = {m:.4f} | fitness = {f:.4f} | accuracy = {acc}/4")
    # Sanity checks (pas stricts)
    assert 0.0 < f < 1.0
    assert 0 <= acc <= 4

# ------------------------------
# 3) Mutations
# ------------------------------
def test_mutations(g: Genome, tracker: InnovationTracker):
    banner("3) Mutations")
    print_genome(g, "Avant mutations")

    # A) Poids
    changed = mutate_weights(g, CFG)
    print(f"\nmutate_weights → {changed} poids potentiellement modifiés.")
    print_genome(g, "Après mutate_weights")

    # B) Ajout de connexion
    edge = mutate_add_connection(g, tracker, CFG)
    print(f"\nmutate_add_connection → nouvelle arête: {edge}")
    print_genome(g, "Après mutate_add_connection")

    # C) Ajout de nœud
    hid = mutate_add_node(g, tracker, CFG)
    print(f"\nmutate_add_node → nouveau hidden id: {hid}")
    print_genome(g, "Après mutate_add_node")

    # Vérifier que le forward tourne toujours
    m = mse(g)
    print(f"\nMSE après mutations = {m:.4f}")

# ------------------------------
# 4) Crossover (recombinaison)
# ------------------------------
def test_crossover(tracker: InnovationTracker):
    banner("4) Crossover")
    # Parent A (simple)
    A = build_simple_genome(tracker)

    # Parent B = A + un hidden via add_node (pour simuler une structure plus complexe)
    B = build_simple_genome(tracker)
    _ = mutate_add_node(B, tracker, CFG)  # insère un hidden en cassant une arête
    # Légère mutation de poids pour différencier
    mutate_weights(B, CFG)

    print_genome(A, "Parent A")
    print_genome(B, "Parent B")

    # Fitness factices (B supposé plus fit ici)
    fitness_A = 0.75
    fitness_B = 0.85

    child = crossover(A, B, fitness_A, fitness_B, disabled_inherit_prob=0.75)
    print_genome(child, "Enfant (crossover)")

    # Forward pour vérifier que ça tourne
    y = child.forward([1.0, 0.0])
    print("y_hat_child(1,0) =", round(y, 4))
    return child

# ------------------------------
# 5) Spéciation (assignation + sharing)
# ------------------------------
def test_speciation(population, fitnesses):
    banner("5) Spéciation")
    spec = Speciator(
        c1=1.0, c2=1.0, c3=0.4,
        delta_threshold=3.0,
        target_species=3,      # optionnel
        adjust_every=5,        # ajuste toutes les 5 générations
        adjust_step=0.1
    )
    species_ids = spec.speciate(population, fitnesses)
    print("Species IDs:", species_ids)
    print("Taille par espèce:", spec.species_sizes())

    # Fitness ajustées (sharing)
    adj = spec.adjusted_fitnesses_per_species()
    for sid, lst in adj.items():
        print(f"Espèce {sid}: fitness ajustées={['{:.3f}'.format(x) for x in lst]}")
    return spec

# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    random.seed(42)

    # 1) Tracker
    tracker = test_innovation_tracker()

    # 2) Génome simple + évaluation
    G = build_simple_genome(tracker)
    test_forward_and_eval(G)

    # 3) Mutations
    test_mutations(G, tracker)

    # 4) Crossover (A x B -> child)
    child = test_crossover(tracker)

    # 5) Spéciation sur une petite population jouet
    #    - on prend: G (muté), child, et 3 variantes de G
    pop = [G, child]
    # Créer 3 autres génomes à partir de G (copies manuelles minimalistes)
    # (Si tu n'as pas encore implémenté Genome.copy(), on reconstruit rapidement)
    G2 = build_simple_genome(tracker); mutate_weights(G2, CFG)
    G3 = build_simple_genome(tracker); mutate_add_node(G3, tracker, CFG)
    G4 = build_simple_genome(tracker); mutate_add_connection(G4, tracker, CFG)
    pop.extend([G2, G3, G4])

    # Fitness brutes fictives (tu peux mettre les vraies via mse->fitness)
    fits = []
    for idx, gi in enumerate(pop, start=1):
        m = mse(gi)
        f = fitness_from_mse(m)
        fits.append(f)
        print(f"[Pop {idx}] mse={m:.4f}  fitness={f:.4f}")

    _ = test_speciation(pop, fits)

    banner("✅ Fin du run_demo : tout a tourné !")
