# src/evolve/crossover.py
import random
from typing import Dict, Tuple, Optional, Iterable

def crossover(parent_a, parent_b, fitness_a: float, fitness_b: float, *,
              disabled_inherit_prob: float = 0.75):
    """
    Recombine deux parents NEAT alignés par innovation number pour produire un enfant.

    Règles NEAT:
      - Gènes appariés (même innovation chez A et B): choisir au hasard l'un des deux.
      - Gènes disjoints/excessifs: hérités du parent le plus fit.
      - Gènes désactivés: si off chez au moins un parent -> off chez l'enfant avec prob ~0.75.

    Hypothèses:
      - parent_a / parent_b ont .nodes (dict[id->NodeGene]) et .conns (dict[(in,out)->ConnectionGene])
      - Les ConnectionGene portent .innovation (int unique par paire (in,out) via InnovationTracker)

    Retour:
      - Un nouvel objet Genome (même classe que les parents).
    """
    # 1) Déterminer le parent le plus fit
    if fitness_a > fitness_b:
        fitter, weaker = parent_a, parent_b
    elif fitness_b > fitness_a:
        fitter, weaker = parent_b, parent_a
    else:
        # égalité: choisir aléatoirement
        fitter, weaker = random.choice([(parent_a, parent_b), (parent_b, parent_a)])

    # 2) Construire des index innovation -> (conn, source_parent)
    #    On passe par une map innovation pour pouvoir aligner proprement
    def by_innovation(genome) -> Dict[int, Tuple]:
        return {c.innovation: c for c in genome.conns.values()}

    A = by_innovation(parent_a)
    B = by_innovation(parent_b)
    all_innovs = sorted(set(A.keys()) | set(B.keys()))

    # 3) Créer l'enfant (même classe Genome que les parents)
    child = type(parent_a)()  # suppose un __init__() sans args
    from ..core.genome import NodeGene, ConnectionGene  # import local

    # Helper: garantir que le nœud existe dans child
    def ensure_node(node_id: int, source_parent) -> None:
        if node_id in child.nodes:
            return
        # Prendre la définition depuis le parent qui le possède (A ou B)
        src_node = source_parent.nodes.get(node_id)
        if src_node is None:
            # Si le parent ne l'a pas, essayer l'autre parent
            other = parent_b if source_parent is parent_a else parent_a
            src_node = other.nodes[node_id]
        child.add_node(NodeGene(src_node.id, src_node.type))

    # 4) Itérer toutes les innovations (appariés/disjoints/excessifs)
    for inn in all_innovs:
        in_A = inn in A
        in_B = inn in B

        if in_A and in_B:
            # Apparié -> choisir l'un des deux gènes
            chosen = random.choice([A[inn], B[inn]])
            other  = B[inn] if chosen is A.get(inn) else A[inn]

            # Héritage de l'état enabled: si l'un est False -> 75% de chances de rester False
            enabled = chosen.enabled and other.enabled
            if (not chosen.enabled) or (not other.enabled):
                if random.random() < disabled_inherit_prob:
                    enabled = False

            # Assurer la présence des nœuds
            ensure_node(chosen.in_node, parent_a if chosen is A.get(inn) else parent_b)
            ensure_node(chosen.out_node, parent_a if chosen is A.get(inn) else parent_b)

            # Ajouter la connexion (éviter les doublons (in,out))
            key = (chosen.in_node, chosen.out_node)
            if key not in child.conns:
                child.add_connection(ConnectionGene(
                    in_node=chosen.in_node,
                    out_node=chosen.out_node,
                    weight=chosen.weight,
                    enabled=enabled,
                    innovation=inn
                ))

        else:
            # Disjoint/Excessif -> hériter du parent le plus fit s'il possède ce gène
            src_map = A if fitter is parent_a else B
            if inn not in src_map:
                continue  # le plus fit ne l'a pas -> on ignore

            chosen = src_map[inn]
            # Assurer la présence des nœuds (prendre depuis le parent le plus fit)
            ensure_node(chosen.in_node, fitter)
            ensure_node(chosen.out_node, fitter)

            # Héritage de l'état enabled (si le gène du plus fit est off, on peut le garder off)
            enabled = chosen.enabled
            # (Optionnel) tu peux aussi appliquer la règle 75% ici si tu veux de la cohérence stricte

            key = (chosen.in_node, chosen.out_node)
            if key not in child.conns:
                child.add_connection(ConnectionGene(
                    in_node=chosen.in_node,
                    out_node=chosen.out_node,
                    weight=chosen.weight,
                    enabled=enabled,
                    innovation=inn
                ))

    return child
