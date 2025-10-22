# viz_network_activations.py
import json, sys, math
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from src.core.genome import Genome

def load_genome(path: str) -> Genome:
    import json
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return Genome.from_dict(blob["genome"])

def topo_order(genome: Genome):
    indeg = {nid: 0 for nid in genome.nodes}
    out_adj = defaultdict(list)
    for (u,v), c in genome.conns.items():
        if not c.enabled: 
            continue
        out_adj[u].append(v)
        indeg[v] += 1
    q = deque([n for n,d in indeg.items() if d==0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order

def layout_layers(genome: Genome):
    """Assigne une 'colonne' par couche topologique pour placer les nœuds joliment."""
    # BFS dans l'ordre topo, layer = 0 pour inputs/bias
    nodes = genome.nodes
    layers = {nid: (0 if nodes[nid].type in ("input","bias") else None) for nid in nodes}
    order = topo_order(genome)

    changed = True
    while changed:
        changed = False
        for (u,v), c in genome.conns.items():
            if not c.enabled: 
                continue
            lu = layers[u]
            lv = layers[v]
            if lu is not None:
                new_lv = (lu + 1) if (lv is None or lv <= lu) else lv
                if layers[v] != new_lv:
                    layers[v] = new_lv
                    changed = True

    # regroupe par layer et attribue des y espacés
    by_layer = defaultdict(list)
    for nid, L in layers.items():
        if L is None: L = 1
        by_layer[L].append(nid)

    pos = {}
    for L, nids in by_layer.items():
        nids = sorted(nids)
        k = len(nids)
        for i, nid in enumerate(nids):
            y = 1.0 if k==1 else i/(k-1)
            pos[nid] = (L, y)
    return pos

def forward_activations(genome: Genome, A: float, B: float):
    """Retourne dict node_id -> activation (après sigmoïde pour hidden/output)."""
    # on s'appuie sur genome.forward mais on veut les valeurs intermédiaires :
    # on réimplémente brièvement pour récupérer tout le vecteur.
    import math
    nodes = genome.nodes
    conns = [c for c in genome.conns.values() if c.enabled]

    incoming = defaultdict(list)
    indeg = {nid: 0 for nid in nodes}
    for c in conns:
        incoming[c.out_node].append(c)
        indeg[c.out_node] += 1

    # init
    val = {}
    for nid, nd in nodes.items():
        if nd.type == "input":
            # on suppose IDs triés; A sera affecté au plus petit id input, B au suivant
            val[nid] = None  # on settra après
        elif nd.type == "bias":
            val[nid] = 1.0
        else:
            val[nid] = 0.0

    # affecter A,B dans l'ordre des inputs
    inputs = sorted([nid for nid, nd in nodes.items() if nd.type == "input"])
    if len(inputs) != 2:
        raise ValueError("Ce visualiseur attend exactement 2 inputs (XOR).")
    val[inputs[0]] = float(A)
    val[inputs[1]] = float(B)

    # topo
    order = topo_order(genome)
    def sigmoid(x): return 1.0/(1.0+math.exp(-x))

    for nid in order:
        t = nodes[nid].type
        if t in ("input","bias"):
            continue
        s = 0.0
        for c in incoming[nid]:
            s += val[c.in_node] * c.weight
        val[nid] = sigmoid(s)
    return val

def draw_network(genome: Genome, activations: dict, ax=None, title=""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,5))
    pos = layout_layers(genome)
    nodes = genome.nodes

    # Edges: thickness = |w|, color sign (+ / -)
    for (u,v), c in genome.conns.items():
        if not c.enabled: 
            continue
        x1, y1 = pos[u]; x2, y2 = pos[v]
        lw = max(0.5, min(5.0, abs(c.weight)*2.5))
        color = "tab:blue" if c.weight >= 0 else "tab:red"
        ax.plot([x1,x2],[y1,y2], color=color, linewidth=lw, alpha=0.7)

    # Nodes: color intensity = activation (0..1), shape by type
    for nid, (x,y) in pos.items():
        nd = nodes[nid]
        a = activations.get(nid, 0.0)
        face = (a, a, a)  # gris -> blanc selon activation
        edge = "k"
        size = 400 if nd.type=="output" else (280 if nd.type=="hidden" else 260)
        marker = "s" if nd.type=="bias" else "o"
        ax.scatter([x],[y], s=size, marker=marker, facecolors=face, edgecolors=edge, zorder=3)
        ax.text(x, y+0.05, f"{nid}:{nd.type[0]}\n{a:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, max(p[0] for p in pos.values())+0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title)
    plt.tight_layout()
    return ax

def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_network_activations.py runs/<run>/best_gen_genXXX.json")
        sys.exit(1)
    g = load_genome(sys.argv[1])

    # Cycle à travers les 4 cas XOR
    cases = [([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0)]
    fig, ax = plt.subplots(figsize=(7,5))

    for (A,B), y in cases:
        acts = forward_activations(g, A, B)
        ax.clear()
        draw_network(g, acts, ax=ax, title=f"Input=({A},{B})  Target={y}  Pred={g.forward([A,B]):.2f}")
        plt.pause(1.2)  # “animation” simple

    print("Appuie sur Close dans la figure pour terminer.")
    plt.show()

if __name__ == "__main__":
    main()
