# viz_animate_training.py
import os, sys, json, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.core.genome import Genome

CHECKPOINT_RE = re.compile(r"best_gen_gen(\d+)\.json$")

def find_checkpoints(run_dir: str):
    files = []
    for name in os.listdir(run_dir):
        m = CHECKPOINT_RE.match(name)
        if m:
            gen = int(m.group(1))
            files.append((gen, os.path.join(run_dir, name)))
    files.sort(key=lambda t: t[0])  # tri par génération
    return files

def load_genome(path: str) -> Genome:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return Genome.from_dict(blob["genome"]), blob.get("meta", {}), blob.get("generation")

def decision_grid(genome: Genome, n=201):
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    Z = np.empty((n, n), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            Z[iy, ix] = genome.forward([float(x), float(y)])
    return xs, ys, Z

def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_animate_training.py runs/<run_dir> [--save mp4|gif] [--fps 10]")
        sys.exit(1)

    run_dir = sys.argv[1]
    save_fmt = None
    fps = 10

    # parse args
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == "--save" and i + 1 < len(sys.argv):
            save_fmt = sys.argv[i + 1].lower()
        if arg == "--fps" and i + 1 < len(sys.argv):
            fps = int(sys.argv[i + 1])

    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        print(f"Aucun checkpoint trouvé dans {run_dir}")
        sys.exit(1)

    # Pré-calcul des grilles pour accélérer l’animation (optionnel)
    # Ici on calcule à la volée pour garder la mémoire légère.

    fig, ax = plt.subplots(figsize=(6, 6))
    im = None
    title = ax.set_title("")

    # Scatter des 4 points XOR (0 = cercle blanc, 1 = noir plein)
    pts = [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
    for x,y,t in pts:
        ax.scatter([x],[y], s=120, marker="o",
                   edgecolors="k", facecolors="none" if t==0 else "k", zorder=3)

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("A"); ax.set_ylabel("B")

    # Pour réutiliser la même image, on prépare une grille fixe de coordonnées
    ngrid = 201
    xs = np.linspace(0, 1, ngrid)
    ys = np.linspace(0, 1, ngrid)

    def init():
        nonlocal im
        dummy = np.zeros((ngrid, ngrid))
        im = ax.imshow(dummy, origin="lower", extent=[0,1,0,1], vmin=0, vmax=1, aspect="equal")
        return [im, title]

    def update(frame_idx):
        gen, path = ckpts[frame_idx]
        g, meta, saved_gen = load_genome(path)

        # calcule la grille
        Z = np.empty((ngrid, ngrid), dtype=float)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                Z[iy, ix] = g.forward([float(x), float(y)])

        im.set_data(Z)
        best_mse = meta.get("best_mse", None)
        best_fit = meta.get("best_fitness", None)
        t = f"Gen {gen} | " \
            f"MSE={best_mse:.4f}  Fit={best_fit:.4f}" if best_mse is not None and best_fit is not None \
            else f"Gen {gen}"
        title.set_text(t)
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(ckpts), init_func=init, blit=False, interval=1000//fps, repeat=True)

    if save_fmt in ("mp4", "gif"):
        out_path = os.path.join(run_dir, f"decision_animation.{save_fmt}")
        print(f"Sauvegarde de l'animation → {out_path}")
        if save_fmt == "mp4":
            anim.save(out_path, fps=fps, dpi=120)  # nécessite ffmpeg
        else:
            anim.save(out_path, fps=fps, dpi=120)  # nécessite ImageMagick
        print("OK.")
    else:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
