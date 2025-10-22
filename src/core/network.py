import math
from typing import Dict, List, Tuple

def sigmoid(x: float) -> float:
    """Activation sigmoïde standard."""
    return 1.0 / (1.0 + math.exp(-x))

def forward_pass(genome, inputs: List[float]) -> float:
    """
    Exécute un passage avant dans le réseau défini par `genome`.
    - `inputs` : valeurs des nœuds d'entrée (ordre = IDs d'inputs triés).
    Hypothèses :
      - Graphe acyclique (feedforward).
      - 0 ou 1 nœud bias (type == "bias"), valeur fixée à 1.0 si présent.
      - Exactement 1 nœud de sortie.
    Retourne la sortie unique (float).
    """
    # --- 1) Collecte des nœuds par type
    nodes_by_id: Dict[int, str] = {nid: n.type for nid, n in genome.nodes.items()}
    input_ids  = sorted([nid for nid, t in nodes_by_id.items() if t == "input"])
    bias_ids   = [nid for nid, t in nodes_by_id.items() if t == "bias"]
    output_ids = [nid for nid, t in nodes_by_id.items() if t == "output"]
    hidden_ids = [nid for nid, t in nodes_by_id.items() if t == "hidden"]

    if len(output_ids) != 1:
        raise ValueError("Le génome doit avoir exactement 1 nœud de sortie.")
    if len(inputs) != len(input_ids):
        raise ValueError(f"Attendu {len(input_ids)} entrées, reçu {len(inputs)}.")

    output_id = output_ids[0]

    # --- 2) Préparer les entrées (valeurs initiales)
    # mapping: node_id -> activation courante (pré-sigmoïde pour non-inputs avant activation)
    values: Dict[int, float] = {}

    # inputs : assignation par ordre d'ID d'entrée trié (convention simple et stable)
    for nid, val in zip(input_ids, inputs):
        values[nid] = float(val)

    # bias (si présent): 1.0
    for nid in bias_ids:
        values[nid] = 1.0

    # init des autres à 0.0
    for nid in genome.nodes.keys():
        if nid not in values:
            values[nid] = 0.0

    # --- 3) Construire les arêtes entrantes actives + indegrees (pour tri topo)
    incoming: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in genome.nodes.keys()}
    indegree: Dict[int, int] = {nid: 0 for nid in genome.nodes.keys()}

    for conn in genome.conns.values():
        if not conn.enabled:
            continue
        u, v, w = conn.in_node, conn.out_node, conn.weight
        if u not in genome.nodes or v not in genome.nodes:
            raise ValueError("Connexion référencant un nœud inexistant.")
        incoming[v].append((u, w))
        indegree[v] += 1

    # --- 4) Tri topologique (Kahn) sur le graphe des connexions actives
    # Les nœuds sans entrées actives démarrent la file (souvent inputs/bias)
    from collections import deque
    q = deque([nid for nid, deg in indegree.items() if deg == 0])
    topo_order: List[int] = []

    while q:
        u = q.popleft()
        topo_order.append(u)
        # "Retirer" u => décrémenter les indegrees des successeurs
        for conn in genome.conns.values():
            if not conn.enabled:
                continue
            if conn.in_node == u:
                v = conn.out_node
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)

    # S'il reste des nœuds avec indegree > 0, il y a un cycle
    if any(deg > 0 for deg in indegree.values()):
        raise ValueError("Le graphe contient un cycle : exécution feedforward impossible.")

    # --- 5) Propagation + activation
    # Convention :
    #  - inputs/bias : valeur déjà fixée, pas d'activation
    #  - hidden/output : somme pondérée des entrées, puis sigmoïde
    for nid in topo_order:
        ntype = nodes_by_id[nid]
        if ntype in ("input", "bias"):
            # Rien à faire : leur valeur est déjà dans `values`
            pass
        else:
            s = 0.0
            for u, w in incoming[nid]:
                s += values[u] * w
            values[nid] = sigmoid(s)

    # --- 6) Sortie unique
    return float(values[output_id])
