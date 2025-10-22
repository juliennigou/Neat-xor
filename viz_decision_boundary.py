# viz_decision_boundary.py
import json, sys, numpy as np
import matplotlib.pyplot as plt

from src.core.genome import Genome

def load_genome(path: str) -> Genome:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return Genome.from_dict(blob["genome"])

def decision_grid(genome: Genome, n=101):
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    Z = np.zeros((n, n), dtype=float)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            Z[i, j] = genome.forward([float(x), float(y)])
    return xs, ys, Z

def plot_decision(genome: Genome, title="Decision surface"):
    xs, ys, Z = decision_grid(genome, n=201)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        Z, origin="lower", extent=[0,1,0,1], vmin=0, vmax=1, aspect="equal"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("sortie réseau (ŷ)")

    # points XOR
    pts = [
        (0,0,0), (0,1,1), (1,0,1), (1,1,0)
    ]
    for x, y, t in pts:
        ax.scatter([x],[y], s=120, marker="o",
                   edgecolors="k", facecolors="none" if t==0 else "k")

    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viz_decision_boundary.py runs/<run>/best_gen_genXXX.json")
        sys.exit(1)
    g = load_genome(sys.argv[1])
    plot_decision(g, title=sys.argv[1])
