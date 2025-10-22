# src/evolve/mutation.py
import math
import random
from typing import Dict, Tuple, List, Optional

# Types d'appoint pour la lisibilité
NodeId = int
EdgeKey = Tuple[int, int]


# ---------- Helpers internes ----------

def _edge_exists(genome, u: NodeId, v: NodeId) -> bool:
    """Retourne True si une connexion (u->v) existe déjà (active ou non)."""
    return (u, v) in genome.conns

def _would_create_cycle(genome, u: NodeId, v: NodeId) -> bool:
    """
    Vérifie si l'ajout de l'arête u->v créerait un cycle.
    Idée: s'il existe déjà un chemin v --> ... --> u, alors u->v fermerait une boucle.
    On fait un DFS à partir de v en suivant les connexions sortantes ACTIVES.
    """
    stack = [v]
    visited = set()

    # Construire rapidement un index "sortants" pour les connexions actives
    out_adj: Dict[NodeId, List[NodeId]] = {}
    for conn in genome.conns.values():
        if not conn.enabled:
            continue
        out_adj.setdefault(conn.in_node, []).append(conn.out_node)

    while stack:
        cur = stack.pop()
        if cur == u:
            return True
        for nxt in out_adj.get(cur, []):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return False

def _random_weight() -> float:
    """Poids initial pour une nouvelle connexion (change la plage si tu préfères)."""
    return random.uniform(-1.0, 1.0)

def _next_hidden_id(genome) -> int:
    """Génère un nouvel ID de nœud (simple: max+1)."""
    return (max(genome.nodes.keys()) + 1) if genome.nodes else 1


# ---------- Mutations principales ----------

def mutate_weights(genome, cfg) -> int:
    """
    Fait varier les poids des connexions EXISTANTES (activées ou non).
    - Avec prob. p_reset_weight : réinitialise complètement le poids (uniforme).
    - Sinon : ajoute une perturbation gaussienne N(0, sigma).
    Retourne le nombre de connexions modifiées.
    """
    p_reset = cfg["mutation"]["p_reset_weight"]
    sigma   = cfg["mutation"]["weight_sigma"]

    changed = 0
    for conn in genome.conns.values():
        # Choix : on mute toutes les connexions (enabled ou non) pour
        # permettre la "résurrection" potentielle via d'autres opérateurs.
        if random.random() < p_reset:
            conn.weight = _random_weight()
        else:
            conn.weight += random.gauss(0.0, sigma)
        changed += 1
    return changed


def mutate_add_connection(genome, innovation_tracker, cfg) -> Optional[EdgeKey]:
    """
    Ajoute UNE nouvelle connexion (u->v) si possible.
    Contraintes :
      - pas de doublon (u,v) déjà présent,
      - pas de cycle (feedforward),
      - sens logique : on n'entre PAS dans un input/bias, et on ne sort PAS d'un output.
    Retourne la clé d'arête ajoutée (u,v) ou None si rien n'a été ajouté.
    """
    # Sépare les IDs par type
    inputs  = [nid for nid, n in genome.nodes.items() if n.type == "input"]
    biases  = [nid for nid, n in genome.nodes.items() if n.type == "bias"]
    hiddens = [nid for nid, n in genome.nodes.items() if n.type == "hidden"]
    outputs = [nid for nid, n in genome.nodes.items() if n.type == "output"]

    # Candidats sources: inputs, bias, hidden (on ne part pas d'un output)
    sources = inputs + biases + hiddens
    # Candidats cibles: hidden, output (on n'entre pas dans input/bias)
    targets = hiddens + outputs

    # Génère toutes les paires possibles
    candidates: List[EdgeKey] = []
    for u in sources:
        for v in targets:
            if u == v:
                continue
            if _edge_exists(genome, u, v):
                continue
            # Évite des directions absurdes (ex: output->hidden ou output->output déjà exclu via sources)
            # Vérifie l'acyclicité potentielle
            if _would_create_cycle(genome, u, v):
                continue
            candidates.append((u, v))

    if not candidates:
        return None

    # Mélange pour injecter du hasard
    random.shuffle(candidates)
    u, v = candidates[0]

    # Crée la connexion
    innovation = innovation_tracker.get_or_create(u, v)
    from ..core.genome import ConnectionGene  # import local pour éviter cycles d'import
    new_conn = ConnectionGene(in_node=u, out_node=v, weight=_random_weight(),
                              enabled=True, innovation=innovation)
    genome.add_connection(new_conn)
    return (u, v)


def mutate_add_node(genome, innovation_tracker, cfg) -> Optional[int]:
    """
    Insère UN nouveau nœud "hidden" en "cassant" une connexion activée existante.
    Étapes :
      1) choisir une connexion activée (u->v, w),
      2) la désactiver,
      3) créer H (nouveau node_id),
      4) créer u->H (poids=1.0, innovation nouvelle),
      5) créer H->v (poids=w, innovation nouvelle).
    Retourne l'ID du nouveau nœud créé, ou None si aucune connexion activée n'existe.
    """
    # Liste des connexions candidates (activées)
    active_edges = [(k, c) for k, c in genome.conns.items() if c.enabled]
    if not active_edges:
        return None

    # Choisir une connexion au hasard
    (u, v), conn = random.choice(active_edges)
    old_weight = conn.weight
    conn.enabled = False  # on "casse" la connexion

    # Crée le nouveau nœud hidden
    from ..core.genome import NodeGene, ConnectionGene  # import local
    
    h_id = _next_hidden_id(genome)
    new_node = NodeGene(h_id, "hidden")
    genome.add_node(new_node)

    # Deux nouvelles connexions
    inn1 = innovation_tracker.get_or_create(u, h_id)
    inn2 = innovation_tracker.get_or_create(h_id, v)

    conn1 = ConnectionGene(in_node=u, out_node=h_id, weight=1.0,    enabled=True, innovation=inn1)
    conn2 = ConnectionGene(in_node=h_id, out_node=v, weight=old_weight, enabled=True, innovation=inn2)

    # Par construction, pas de cycle (on remplace u->v par u->H->v)
    genome.add_connection(conn1)
    genome.add_connection(conn2)

    return h_id
