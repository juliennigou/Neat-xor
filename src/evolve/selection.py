# src/evolve/selection.py
import math
import random
from typing import Dict, List, Tuple

def _tournament_select(candidates: List[Tuple[object, float]], k: int = 3) -> object:
    """Sélectionne 1 parent via tournoi (plus fit favorisé)."""
    picks = random.sample(candidates, k=min(k, len(candidates)))
    picks.sort(key=lambda t: t[1], reverse=True)  # tri par fitness décroissante
    return picks[0][0]  # retourne le genome du meilleur pick

def allocate_offspring_per_species(species_list, population_size: int) -> Dict[int, int]:
    """
    Quote-part d'enfants par espèce, proportionnelle à la somme des fitness ajustées.
    Retour: {species_id: n_offspring}
    """
    sums = []
    for sp in species_list:
        adj = sp.adjusted_fitnesses()
        sums.append((sp.id, sum(adj)))

    total = sum(s for _, s in sums)
    if total <= 0.0:
        # Répartition uniforme si tout est nul
        base = population_size // len(species_list)
        rest = population_size - base * len(species_list)
        alloc = {sp.id: base for sp in species_list}
        # distribue le reste
        for sp in species_list[:rest]:
            alloc[sp.id] += 1
        return alloc

    # allocation proportionnelle
    raw = [(sid, population_size * s / total) for sid, s in sums]
    alloc = {sid: int(x) for sid, x in raw}
    # corriger les arrondis pour atteindre exactement population_size
    diff = population_size - sum(alloc.values())
    # donne le diff aux plus gros résidus
    residuals = sorted(((sid, x - alloc[sid]) for sid, x in raw),
                       key=lambda t: t[1], reverse=True)
    for i in range(diff):
        alloc[residuals[i % len(residuals)][0]] += 1
    return alloc

def reproduce_species(
    sp,
    n_children: int,
    *,
    elitism_per_species: int,
    population_fitness_map: Dict[object, float],
    crossover_fn,           # (pa, pb, fa, fb) -> child_genome
    mutate_weight_fn,       # (genome, cfg) -> int
    mutate_add_conn_fn,     # (genome, tracker, cfg) -> (u,v) or None
    mutate_add_node_fn,     # (genome, tracker, cfg) -> node_id or None
    tracker,
    cfg
) -> List[object]:
    """
    Produit n_children individus pour cette espèce.
    - Copie les élites (meilleurs) tels quels (jusqu'à elitism_per_species, mais <= n_children).
    - Le reste via crossover/mutations.
    """
    members = sp.members[:]           # [(genome, fitness), ...]
    members.sort(key=lambda t: t[1], reverse=True)
    genomes_only = [g for (g, _) in members]

    offspring: List[object] = []

    # 1) Élites
    elites = min(elitism_per_species, n_children, len(genomes_only))
    for i in range(elites):
        elite_parent = genomes_only[i]
        child = elite_parent.copy()
        offspring.append(child)

    remaining = n_children - len(offspring)
    if remaining <= 0:
        return offspring

    # 2) Générer le reste
    # Probabilité : 75% crossover, 25% mutation seule (ajuste si tu veux via cfg)
    p_crossover = 0.75

    # pool pour tournoi (genome, fitness)
    pool = members

    from ..evolve.crossover import crossover as default_crossover
    crossover_impl = crossover_fn or default_crossover

    for _ in range(remaining):
        if random.random() < p_crossover and len(pool) >= 2:
            # sélection de 2 parents (tournoi k=3)
            pa = _tournament_select(pool, k=3)
            pb = _tournament_select(pool, k=3)
            fa = population_fitness_map[pa]
            fb = population_fitness_map[pb]
            child = crossover_impl(pa, pb, fa, fb, disabled_inherit_prob=0.75)
        else:
            # mutation seule: pick un parent et copie-le
            pa = _tournament_select(pool, k=3)
            child = pa.copy()


        # Appliquer mutations (probas côté loop)
        if random.random() < cfg["mutation"].get("p_mutate_weight", 0.8):
            mutate_weight_fn(child, cfg)
        if random.random() < cfg["mutation"].get("p_add_conn", 0.05):
            mutate_add_conn_fn(child, tracker, cfg)
        if random.random() < cfg["mutation"].get("p_add_node", 0.03):
            mutate_add_node_fn(child, tracker, cfg)

        offspring.append(child)

    return offspring
