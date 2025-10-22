# viz_genome.py (extrait à coller dans ton projet, p.ex. à côté de genome.py)
from __future__ import annotations
import math
from typing import Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt

# --- helpers de layout : positions "couchées" par type ---
def _layered_positions(nodes: Dict[int, "NodeGene"]) -> Dict[int, Tuple[float, float]]:
    """
    Place les nœuds en colonnes : input/bias | hidden | output
    Retourne un dict {node_id: (x, y)}.
    """
    cols = {"input": 0, "bias": 0, "hidden": 1, "output": 2}
    buckets: Dict[str, list[int]] = {"input": [], "bias": [], "hidden": [], "output": []}
    for n in nodes.values():
        buckets[n.type].append(n.id)

    # Ordonne pour stabilité
    for k in buckets:
        buckets[k].sort()

    pos: Dict[int, Tuple[float, float]] = {}
    for t, x in cols.items():
        ids = buckets[t]
        if not ids:
            continue
        # espace vertical régulier
        h = len(ids)
        for i, nid in enumerate(ids):
            # centre vertical sur 0 (haut = positif)
            y = (h - 1) / 2.0 - i
            pos[nid] = (float(x), float(y))
    return pos

def genome_to_networkx(genome: "Genome") -> nx.DiGraph:
    """Convertit le Genome en DiGraph NetworkX avec attributs utiles."""
    G = nx.DiGraph()
    # Nœuds
    for node in genome.nodes.values():
        G.add_node(node.id, type=node.type)
    # Connexions
    for c in genome.conns.values():
        G.add_edge(
            c.in_node,
            c.out_node,
            weight=c.weight,
            innovation=c.innovation,
            enabled=bool(c.enabled),
        )
    return G

def visualize_genome(genome: "Genome", layout: str = "layered", figsize=(7, 5)) -> None:
    """
    Affiche le génome :
    - layout "layered" (défaut) ou "spring"
    - largeur d'arête proportionnelle à |poids|
    - arêtes désactivées en gris pointillé
    """
    G = genome_to_networkx(genome)

    if layout == "layered":
        pos = _layered_positions(genome.nodes)
    else:
        # layout ressort générique si tu préfères
        pos = nx.spring_layout(G, seed=42)

    # Couleurs par type
    color_map = {
        "input":  "#4C78A8",
        "bias":   "#66A61E",
        "hidden": "#A6CEE3",
        "output": "#E45756",
    }
    # Dessine les nœuds par type (pour formes/couleurs cohérentes)
    for t, shape in [("input", "o"), ("bias", "s"), ("hidden", "o"), ("output", "D")]:
        nodes_t = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        if not nodes_t:
            continue
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes_t,
            node_color=color_map[t],
            node_shape=shape,
            node_size=800,
            linewidths=1.0,
            edgecolors="black",
        )

    # Labels de nœuds (id)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()}, font_size=9)

    # Sépare arêtes activées/désactivées
    enabled_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("enabled", True)]
    disabled_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("enabled", True)]

    # Largeurs proportionnelles |poids|
    def widths(edges):
        return [1.0 + 2.0 * abs(G[u][v].get("weight", 0.0)) for u, v in edges]

    # Arêtes activées
    nx.draw_networkx_edges(
        G, pos,
        edgelist=enabled_edges,
        width=widths(enabled_edges),
        arrows=True,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
    )
    # Arêtes désactivées
    nx.draw_networkx_edges(
        G, pos,
        edgelist=disabled_edges,
        width=widths(disabled_edges),
        style="dashed",
        edge_color="#999999",
        arrows=True,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
    )

    # Labels d'arêtes : poids (3 déc.)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Mise en page
    plt.figure(figsize=figsize)
    plt.axis("off")
    # Petite astuce : NetworkX dessine sur la figure active ; on force un redraw
    plt.tight_layout()
    plt.show()
