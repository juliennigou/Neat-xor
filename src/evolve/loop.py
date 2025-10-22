# src/evolve/loop.py
from typing import Dict, List, Tuple
from ..eval.xor_task import mse, fitness_from_mse
from .selection import allocate_offspring_per_species, reproduce_species

from ..io.checkpoints import save_best
from ..io.logging_io import CsvLogger
import os
import time


def evaluate_population(population) -> Tuple[List[float], List[float]]:
    """Retourne (mses, fitnesses) alignés à la population."""
    mses = [mse(g) for g in population]
    fits = [fitness_from_mse(m) for m in mses]
    return mses, fits

def evolve_one_generation(
    population: List[object],
    tracker,
    speciator,
    cfg,
    *,
    crossover_fn=None,
    mutate_weight_fn=None,
    mutate_add_conn_fn=None,
    mutate_add_node_fn=None
) -> Tuple[List[object], Dict]:
    """
    Fait une génération: évaluation -> spéciation -> reproduction -> nouvelle population.
    Retourne (new_population, logs).
    """
    # 1) Évaluer
    mses, fits = evaluate_population(population)
    best_idx = min(range(len(mses)), key=lambda i: mses[i])
    best_genome = population[best_idx]
    best_mse = mses[best_idx]

    # 2) Spéciation
    species_ids = speciator.speciate(population, fits)

    # 3) Mapping fitness
    pop_fit_map = {g: f for g, f in zip(population, fits)}

    # 4) Quotas par espèce
    pop_size = len(population)
    alloc = allocate_offspring_per_species(speciator.species, pop_size)

    # 5) Reproduction par espèce
    new_pop: List[object] = []
    for sp in speciator.species:
        n_kids = alloc.get(sp.id, 0)
        if n_kids <= 0:
            continue
        kids = reproduce_species(
            sp,
            n_kids,
            elitism_per_species=cfg.get("elitism_per_species", 1),
            population_fitness_map=pop_fit_map,
            crossover_fn=crossover_fn,
            mutate_weight_fn=mutate_weight_fn,
            mutate_add_conn_fn=mutate_add_conn_fn,
            mutate_add_node_fn=mutate_add_node_fn,
            tracker=tracker,
            cfg=cfg
        )
        new_pop.extend(kids)

    # 6) Ajuster la taille (au cas où)
    if len(new_pop) > pop_size:
        new_pop = new_pop[:pop_size]
    elif len(new_pop) < pop_size:
        # compléter en dupliquant quelques meilleurs de l'ancienne pop (fallback)
        best_indices = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
        for i in best_indices:
            if len(new_pop) >= pop_size:
                break
            new_pop.append(population[i])

    # 7) Logs
    logs = {
        "best_genome": best_genome,
        "best_mse": best_mse,
        "mean_mse": sum(mses) / len(mses),
        "best_fitness": max(fits),
        "mean_fitness": sum(fits) / len(fits),
        "n_species": len(speciator.species),
        "species_sizes": [len(sp.members) for sp in speciator.species],
        "delta_threshold": speciator.delta_threshold,
    }
    return new_pop, logs

def run_training_loop(
    make_initial_population,   # () -> List[Genome]
    tracker,
    speciator,
    cfg: dict,
    max_generations: int = 300,
    success_mse: float = 0.01,
    *,
    crossover_fn=None,
    mutate_weight_fn=None,
    mutate_add_conn_fn=None,
    mutate_add_node_fn=None,
    verbose: bool = True,
    run_name: str = "xor"
) -> Tuple[List[object], list]:
    """
    Boucle d'entraînement complète NEAT (version finale).
    - Crée la population initiale
    - Répète: évaluer -> spéciation -> reproduction -> nouvelle génération
    - Sauvegarde le meilleur génome (checkpoint) et loggue les métriques en CSV

    Retourne (population_finale, history_logs).
    """
    # --- Dossier de run + logger CSV
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{run_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = CsvLogger(
        os.path.join(run_dir, "metrics.csv"),
        fieldnames=[
            "generation", "best_mse", "mean_mse",
            "best_fitness", "mean_fitness",
            "n_species", "species_sizes", "delta_threshold",
        ],
    )

    # --- Population initiale
    population = make_initial_population()

    # Pour suivi du meilleur global et checkpoints
    best_so_far = float("inf")
    history = []

    for gen in range(1, max_generations + 1):
        # 1) Une génération d’évolution
        population, logs = evolve_one_generation(
            population, tracker, speciator, cfg,
            crossover_fn=crossover_fn,
            mutate_weight_fn=mutate_weight_fn,
            mutate_add_conn_fn=mutate_add_conn_fn,
            mutate_add_node_fn=mutate_add_node_fn
        )

        # 2) Ré-évaluer la population courante pour extraire le meilleur exact (idx & mse)
        mses, fits = evaluate_population(population)
        best_idx = min(range(len(mses)), key=lambda i: mses[i])
        best_mse = mses[best_idx]
        best_fit = fits[best_idx]

        # 3) Checkpoint si amélioration
        if best_mse < best_so_far:
            best_so_far = best_mse
            save_best(population[best_idx], run_dir, gen, extra={
                "best_mse": best_so_far,
                "best_fitness": best_fit
            })

        # 4) Logging CSV
        logger.write_row({
            "generation": gen,
            "best_mse": best_mse,
            "mean_mse": logs["mean_mse"],
            "best_fitness": logs["best_fitness"],
            "mean_fitness": logs["mean_fitness"],
            "n_species": logs["n_species"],
            "species_sizes": "|".join(map(str, logs["species_sizes"])),
            "delta_threshold": logs["delta_threshold"],
        })

        # 5) Impression console
        if verbose:
            print(f"[Gen {gen:03d}] "
                  f"best_mse={best_mse:.4f}  mean_mse={logs['mean_mse']:.4f}  "
                  f"best_fit={best_fit:.4f}  species={logs['n_species']}  "
                  f"sizes={logs['species_sizes']}  δ_t={logs['delta_threshold']:.2f}")

        # 6) Critère d'arrêt
        if best_mse <= success_mse:
            if verbose:
                print(f"✅ Succès atteint à la génération {gen} (best_mse={best_mse:.6f})")
            break

    return population, history