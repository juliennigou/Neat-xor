# src/evolve/speciation.py
from typing import Dict, List, Tuple, Optional
import random

def compute_compatibility_distance(genome_a, genome_b, *, c1: float, c2: float, c3: float) -> float:
    """
    Calcule δ entre deux génomes NEAT (connexions alignées par innovation).
    Hypothèses:
      - genome.*.conns: dict[(in,out)->ConnectionGene] avec attribut .innovation et .weight
    """
    # Index par innovation
    A = {c.innovation: c for c in genome_a.conns.values()}
    B = {c.innovation: c for c in genome_b.conns.values()}

    if not A and not B:
        return 0.0

    all_innovs = sorted(set(A.keys()) | set(B.keys()))
    max_innov_common = max(set(A.keys()) & set(B.keys())) if (set(A.keys()) & set(B.keys())) else -1

    # Compteurs
    matched = 0
    weight_diff_sum = 0.0
    disjoint = 0
    excess = 0

    for inn in all_innovs:
        inA = inn in A
        inB = inn in B
        if inA and inB:
            matched += 1
            weight_diff_sum += abs(A[inn].weight - B[inn].weight)
        else:
            # disjoint vs excess : dépend si inn est > max_innov_common
            if inn > max_innov_common and max_innov_common != -1:
                excess += 1
            else:
                disjoint += 1

    Wbar = (weight_diff_sum / matched) if matched > 0 else 0.0
    N = max(1, max(len(A), len(B)))
    delta = (c1 * excess) / N + (c2 * disjoint) / N + (c3 * Wbar)
    return float(delta)


class Species:
    """
    Représente une espèce NEAT.
    - representative: un génome "exemple" (sert à l'assignation)
    - members: liste de tuples (genome, fitness_brute)
    """
    _id_counter = 1

    def __init__(self, representative):
        self.id = Species._id_counter
        Species._id_counter += 1

        self.representative = representative
        self.members: List[Tuple[object, float]] = []  # (genome, fitness)
        self.age = 0
        self.best_fitness = float("-inf")

    def reset_members(self):
        self.members.clear()

    def add_member(self, genome, fitness: float):
        self.members.append((genome, fitness))

    def update_representative(self):
        # Option simple: choisir un membre au hasard
        if self.members:
            self.representative = random.choice(self.members)[0]

    def adjusted_fitnesses(self) -> List[float]:
        """
        Fitness sharing: fitness / taille_de_l_espece
        (version simple: facteur constant = 1/|S|)
        """
        if not self.members:
            return []
        denom = len(self.members)
        return [fit / denom for (_, fit) in self.members]

    def mean_adjusted_fitness(self) -> float:
        af = self.adjusted_fitnesses()
        return sum(af) / len(af) if af else 0.0


class Speciator:
    """
    Gère l'assignation des génomes aux espèces et (optionnel) l'ajustement du delta_threshold.
    """
    def __init__(self, *, c1: float, c2: float, c3: float,
                 delta_threshold: float,
                 target_species: Optional[int] = None,
                 adjust_every: int = 10,
                 adjust_step: float = 0.1):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.delta_threshold = delta_threshold
        self.target_species = target_species
        self.adjust_every = adjust_every
        self.adjust_step = adjust_step

        self.species: List[Species] = []
        self.generation = 0

    def _distance(self, a, b) -> float:
        return compute_compatibility_distance(a, b, c1=self.c1, c2=self.c2, c3=self.c3)

    def speciate(self, population: List[object], fitnesses: List[float]) -> List[int]:
        """
        Assigne chaque génome à une espèce.
        Retourne une liste species_ids alignée avec `population`.
        """
        assert len(population) == len(fitnesses)
        self.generation += 1

        # 1) Réinitialiser les membres
        for sp in self.species:
            sp.reset_members()

        # 2) Assigner chaque génome à la PREMIERE espèce dont δ ≤ seuil,
        #    sinon créer une nouvelle espèce.
        species_ids: List[int] = []
        for g, fit in zip(population, fitnesses):
            assigned_id = None
            # Essayer chaque espèce existante
            for sp in self.species:
                d = self._distance(g, sp.representative)
                if d <= self.delta_threshold:
                    sp.add_member(g, fit)
                    assigned_id = sp.id
                    break
            # Si non assigné -> nouvelle espèce
            if assigned_id is None:
                new_sp = Species(representative=g)
                new_sp.add_member(g, fit)
                self.species.append(new_sp)
                assigned_id = new_sp.id
            species_ids.append(assigned_id)

        # 3) Mettre à jour les représentants (optionnel: au hasard)
        for sp in self.species:
            sp.update_representative()
            # best fitness (utile pour stagnation plus tard)
            if sp.members:
                sp.best_fitness = max(sp.best_fitness, max(f for _, f in sp.members))
            sp.age += 1

        # 4) Option: ajuster le seuil pour viser target_species
        if self.target_species and (self.generation % self.adjust_every == 0):
            n = len(self.species)
            if n > self.target_species:
                self.delta_threshold += self.adjust_step
            elif n < self.target_species:
                self.delta_threshold = max(0.0, self.delta_threshold - self.adjust_step)

        return species_ids

    def adjusted_fitnesses_per_species(self) -> Dict[int, List[float]]:
        """
        Retourne: {species_id: [fitness_adjusted_i, ...]}
        """
        res: Dict[int, List[float]] = {}
        for sp in self.species:
            res[sp.id] = sp.adjusted_fitnesses()
        return res

    def species_sizes(self) -> Dict[int, int]:
        return {sp.id: len(sp.members) for sp in self.species}

    def prune_empty(self):
        """Enlève les espèces vides (peut arriver selon la logique de boucle)."""
        self.species = [sp for sp in self.species if sp.members]
