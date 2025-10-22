import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm, colors, colormaps
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from train_xor import make_initial_population_factory
from src.core.genome import Genome
from src.core.innovation import InnovationTracker
from src.evolve.crossover import crossover
from src.evolve.loop import evaluate_population
from src.evolve.mutation import mutate_add_connection, mutate_add_node, mutate_weights
from src.evolve.selection import allocate_offspring_per_species, reproduce_species
from src.evolve.speciation import Speciator


XOR_CASES: Sequence[Tuple[Tuple[float, float], float]] = (
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
)

SPECIES_MARKERS = ("o", "s", "^", "D", "P", "X", "*", "v")

TYPE_STYLES: Dict[str, Dict[str, object]] = {
    "input": {"shape": "^", "edge": "#1f77b4", "size": 600, "lw": 1.4, "label": "Entrée"},
    "bias": {"shape": "s", "edge": "#ff7f0e", "size": 520, "lw": 1.6, "label": "Biais"},
    "hidden": {"shape": "o", "edge": "#6c757d", "size": 500, "lw": 1.2, "label": "Hidden"},
    "output": {"shape": "D", "edge": "#2ca02c", "size": 650, "lw": 1.8, "label": "Sortie"},
}


@dataclass
class GenerationSnapshot:
    generation: int
    population: List[Genome]
    mses: List[float]
    fitnesses: List[float]
    species_ids: List[int]
    best_index: int
    best_mse: float
    mean_mse: float


def compute_node_layers(genome: Genome) -> Dict[int, int]:
    incoming: Dict[int, List[int]] = {nid: [] for nid in genome.nodes.keys()}
    for conn in genome.conns.values():
        if not conn.enabled:
            continue
        incoming.setdefault(conn.out_node, []).append(conn.in_node)

    layers: Dict[int, int] = {}
    base_nodes = {
        nid for nid, node in genome.nodes.items() if node.type in ("input", "bias")
    }
    for nid in base_nodes:
        layers[nid] = 0

    def assign(node_id: int) -> int:
        if node_id in layers:
            return layers[node_id]
        parents = incoming.get(node_id, [])
        if not parents:
            layers[node_id] = 1
            return 1
        depth = 1 + max(assign(pid) for pid in parents)
        layers[node_id] = depth
        return depth

    for nid in genome.nodes.keys():
        assign(nid)
    return layers


def compute_activations(genome: Genome, inputs: Sequence[float]) -> Dict[int, float]:
    nodes_by_id = {nid: n.type for nid, n in genome.nodes.items()}
    input_ids = sorted([nid for nid, t in nodes_by_id.items() if t == "input"])
    bias_ids = [nid for nid, t in nodes_by_id.items() if t == "bias"]
    hidden_ids = [nid for nid, t in nodes_by_id.items() if t == "hidden"]
    output_ids = [nid for nid, t in nodes_by_id.items() if t == "output"]

    if len(output_ids) != 1:
        raise ValueError("Le génome doit avoir exactement 1 nœud de sortie.")
    if len(inputs) != len(input_ids):
        raise ValueError(f"Attendu {len(input_ids)} entrées, reçu {len(inputs)}.")

    values: Dict[int, float] = {}
    for nid, val in zip(input_ids, inputs):
        values[nid] = float(val)
    for nid in bias_ids:
        values[nid] = 1.0
    for nid in genome.nodes.keys():
        values.setdefault(nid, 0.0)

    incoming: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in genome.nodes.keys()}
    indegree: Dict[int, int] = {nid: 0 for nid in genome.nodes.keys()}
    for conn in genome.conns.values():
        if not conn.enabled:
            continue
        incoming[conn.out_node].append((conn.in_node, conn.weight))
        indegree[conn.out_node] += 1

    from collections import deque

    queue = deque([nid for nid, deg in indegree.items() if deg == 0])
    topo_order: List[int] = []
    while queue:
        nid = queue.popleft()
        topo_order.append(nid)
        for conn in genome.conns.values():
            if not conn.enabled:
                continue
            if conn.in_node == nid:
                indegree[conn.out_node] -= 1
                if indegree[conn.out_node] == 0:
                    queue.append(conn.out_node)

    if any(deg > 0 for deg in indegree.values()):
        raise ValueError("Cycle détecté dans le génome; impossible de propager.")

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    for nid in topo_order:
        ntype = nodes_by_id[nid]
        if ntype in ("input", "bias"):
            continue
        total = 0.0
        for src, weight in incoming[nid]:
            total += values[src] * weight
        values[nid] = sigmoid(total)

    return values


class NeatLiveVisualizer:
    def __init__(
        self,
        population: List[Genome],
        tracker: InnovationTracker,
        speciator: Speciator,
        cfg: Dict,
        *,
        max_generations: int,
        success_mse: float,
        seed: Optional[int] = None,
        verbose: bool = False,
        save_gif_path: Optional[str] = None,
        gif_fps: int = 10,
    ) -> None:
        self.population = population
        self.tracker = tracker
        self.speciator = speciator
        self.cfg = cfg
        self.max_generations = max_generations
        self.success_mse = success_mse
        self.verbose = verbose
        self.save_gif_path = save_gif_path
        self.gif_fps = gif_fps

        self.rng = np.random.default_rng(seed)
        self.selected_index: Optional[int] = None
        self.rank_cursor = 0
        self.case_index = 0
        self.paused = False
        self.running = True

        self.current_snapshot: Optional[GenerationSnapshot] = None
        self.scatter_items: List[Tuple[PathCollection, List[int]]] = []
        self._captured_frames: List[np.ndarray] = []

        self.fig = plt.figure(figsize=(13, 6))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])
        self.ax_ecosystem = self.fig.add_subplot(gs[:, 0])
        self.ax_network = self.fig.add_subplot(gs[0, 1])
        self.ax_surface = self.fig.add_subplot(gs[1, 1])

        self.cmap = colormaps["RdYlGn_r"]
        self.node_cmap = colormaps["Blues"]
        self.norm = colors.Normalize(vmin=0.0, vmax=0.25)
        self.scalar_mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.colorbar = self.fig.colorbar(
            self.scalar_mappable,
            ax=self.ax_ecosystem,
            orientation="vertical",
            pad=0.02,
        )
        self.colorbar.set_label("MSE")

        self.grid_n = 101
        self.grid_x = np.linspace(0, 1, self.grid_n)
        self.grid_y = np.linspace(0, 1, self.grid_n)
        self.surface_im = self.ax_surface.imshow(
            np.zeros((self.grid_n, self.grid_n)),
            origin="lower",
            extent=[0, 1, 0, 1],
            vmin=0,
            vmax=1,
            cmap="viridis",
            aspect="equal",
        )
        self.surface_contour = None

        xor_points = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
        for x, y, target in xor_points:
            if target == 1:
                self.ax_surface.scatter([x], [y], s=120, color="k", marker="o", zorder=3)
            else:
                self.ax_surface.scatter(
                    [x], [y], s=120, edgecolors="k", facecolors="none", marker="o", zorder=3
                )
        self.case_marker = self.ax_surface.scatter(
            [0],
            [0],
            s=250,
            edgecolors="cyan",
            facecolors="none",
            linewidths=2.0,
            zorder=4,
        )

        self.ax_surface.set_xlim(0, 1)
        self.ax_surface.set_ylim(0, 1)
        self.ax_surface.set_xlabel("A")
        self.ax_surface.set_ylabel("B")
        self.ax_surface.set_title("Surface de décision (ŷ)")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def run(self) -> None:
        for gen in range(self.max_generations + 1):
            if not self.running:
                break
            snapshot = self.evaluate_generation(gen)
            self.current_snapshot = snapshot
            self.update_visuals(snapshot)
            if self.verbose:
                print(
                    f"[Gen {snapshot.generation:03d}] "
                    f"best_mse={snapshot.best_mse:.4f}  mean_mse={snapshot.mean_mse:.4f}"
                )
            if snapshot.best_mse <= self.success_mse:
                break
            while self.paused and self.running:
                plt.pause(0.05)
            if not self.running:
                break
            self.population = self.make_next_population(snapshot)
            plt.pause(0.01)
        self.export_gif_if_needed()
        if self.running:
            plt.show()

    def evaluate_generation(self, gen_number: int) -> GenerationSnapshot:
        mses, fits = evaluate_population(self.population)
        species_ids = self.speciator.speciate(self.population, fits)
        best_index = min(range(len(mses)), key=lambda i: mses[i])
        best_mse = mses[best_index]
        mean_mse = float(np.mean(mses)) if mses else float("nan")
        return GenerationSnapshot(
            generation=gen_number,
            population=list(self.population),
            mses=mses,
            fitnesses=fits,
            species_ids=species_ids,
            best_index=best_index,
            best_mse=best_mse,
            mean_mse=mean_mse,
        )

    def make_next_population(self, snapshot: GenerationSnapshot) -> List[Genome]:
        pop_size = len(snapshot.population)
        pop_fit_map = {g: fit for g, fit in zip(snapshot.population, snapshot.fitnesses)}
        alloc = allocate_offspring_per_species(self.speciator.species, pop_size)
        new_population: List[Genome] = []
        for species in self.speciator.species:
            n_children = alloc.get(species.id, 0)
            if n_children <= 0:
                continue
            kids = reproduce_species(
                species,
                n_children,
                elitism_per_species=self.cfg.get("elitism_per_species", 1),
                population_fitness_map=pop_fit_map,
                crossover_fn=crossover,
                mutate_weight_fn=mutate_weights,
                mutate_add_conn_fn=mutate_add_connection,
                mutate_add_node_fn=mutate_add_node,
                tracker=self.tracker,
                cfg=self.cfg,
            )
            new_population.extend(kids)
        if len(new_population) > pop_size:
            new_population = new_population[:pop_size]
        elif len(new_population) < pop_size:
            best_indices = sorted(
                range(len(snapshot.fitnesses)),
                key=lambda i: snapshot.fitnesses[i],
                reverse=True,
            )
            for idx in best_indices:
                if len(new_population) >= pop_size:
                    break
                new_population.append(snapshot.population[idx].copy())
        return new_population

    def update_visuals(self, snapshot: GenerationSnapshot) -> None:
        self.rank_cursor = 0
        self.ax_ecosystem.clear()
        self.ax_ecosystem.set_xlim(0, 1)
        self.ax_ecosystem.set_ylim(0, 1)
        self.set_ecosystem_title(snapshot)
        self.draw_species_panel(snapshot)
        self.selected_index = snapshot.best_index
        self.update_network_and_surface()
        self.fig.canvas.draw_idle()
        self.capture_frame_if_needed()

    def draw_species_panel(self, snapshot: GenerationSnapshot) -> None:
        mses = snapshot.mses
        if not mses:
            return
        species_to_members: Dict[int, List[int]] = {}
        for idx, sid in enumerate(snapshot.species_ids):
            species_to_members.setdefault(sid, []).append(idx)

        centers = {
            sid: (
                float(self.rng.uniform(0.1, 0.9)),
                float(self.rng.uniform(0.1, 0.9)),
            )
            for sid in species_to_members.keys()
        }
        sigma = 0.07
        self.scatter_items.clear()

        legend_handles: List[Line2D] = []
        best_idx = snapshot.best_index
        for offset, (sid, member_indices) in enumerate(species_to_members.items()):
            cx, cy = centers[sid]
            xs = []
            ys = []
            colors_for_members = []
            sizes = []
            for idx in member_indices:
                x = float(self.rng.normal(cx, sigma))
                y = float(self.rng.normal(cy, sigma))
                xs.append(np.clip(x, 0.0, 1.0))
                ys.append(np.clip(y, 0.0, 1.0))
                colors_for_members.append(self.cmap(self.norm(snapshot.mses[idx])))
                size = 140 if idx == best_idx else 80
                sizes.append(size)
            marker = SPECIES_MARKERS[offset % len(SPECIES_MARKERS)]
            scatter = self.ax_ecosystem.scatter(
                xs,
                ys,
                s=sizes,
                marker=marker,
                c=colors_for_members,
                edgecolors="k",
                linewidths=0.5,
                picker=True,
            )
            self.scatter_items.append((scatter, member_indices))
            legend_handles.append(Line2D([], [], color="k", marker=marker, linestyle="", label=f"S{sid}"))

        if legend_handles:
            self.ax_ecosystem.legend(handles=legend_handles[:8], title="Espèces", loc="upper right")
        self.ax_ecosystem.text(
            0.02,
            0.02,
            f"{len(species_to_members)} espèces",
            transform=self.ax_ecosystem.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    def update_network_and_surface(self) -> None:
        if not self.current_snapshot or self.selected_index is None:
            return
        snapshot = self.current_snapshot
        best_idx = snapshot.best_index
        genome = snapshot.population[best_idx]
        inputs, target = XOR_CASES[self.case_index]
        activations = compute_activations(genome, inputs)
        pred = activations[
            next(nid for nid, node in genome.nodes.items() if node.type == "output")
        ]
        self.draw_network_panel(genome, activations, inputs, target, pred)
        self.draw_surface_panel(genome, inputs)

    def draw_network_panel(
        self,
        genome: Genome,
        activations: Dict[int, float],
        inputs: Tuple[float, float],
        target: float,
        prediction: float,
    ) -> None:
        self.ax_network.clear()
        self.ax_network.set_title(
            f"Cas XOR {inputs} → cible={target:.1f}, ŷ={prediction:.3f}"
        )
        layers = compute_node_layers(genome)
        max_layer = max(layers.values()) if layers else 1
        per_layer: Dict[int, List[int]] = {}
        for nid, layer in layers.items():
            per_layer.setdefault(layer, []).append(nid)

        positions: Dict[int, Tuple[float, float]] = {}
        for layer, node_ids in per_layer.items():
            node_ids.sort()
            y_positions = np.linspace(0.9, 0.1, num=len(node_ids)) if node_ids else []
            for idx, nid in enumerate(node_ids):
                x = 0.1 + 0.8 * (layer / max_layer if max_layer else 0.5)
                y = float(y_positions[idx]) if len(node_ids) > 1 else 0.5
                positions[nid] = (x, y)

        graph = nx.DiGraph()
        for nid, node in genome.nodes.items():
            graph.add_node(nid, type=node.type)
        for conn in genome.conns.values():
            if conn.enabled:
                graph.add_edge(conn.in_node, conn.out_node, weight=conn.weight)

        type_to_nodes: Dict[str, List[int]] = {}
        for nid, data in graph.nodes(data=True):
            type_to_nodes.setdefault(data["type"], []).append(nid)

        legend_handles: List[Line2D] = []
        for node_type, nodelist in type_to_nodes.items():
            if not nodelist:
                continue
            style = TYPE_STYLES.get(node_type, TYPE_STYLES["hidden"])
            shape = style["shape"]
            edge_color = style["edge"]
            size = style["size"]
            lw = style["lw"]
            colors_subset = [self.node_cmap(activations[nid]) for nid in nodelist]
            sizes = [size] * len(nodelist)
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=nodelist,
                node_shape=shape,
                node_color=colors_subset,
                edgecolors=edge_color,
                linewidths=lw,
                node_size=sizes,
                ax=self.ax_network,
            )
            legend_handles.append(
                Line2D(
                    [], [],
                    marker=shape,
                    linestyle="",
                    markerfacecolor="#dddddd",
                    markeredgecolor=edge_color,
                    markeredgewidth=lw,
                    markersize=9,
                    label=style["label"],
                )
            )

        edge_colors = []
        widths = []
        for u, v, data in graph.edges(data=True):
            w = data["weight"]
            edge_colors.append("tab:blue" if w >= 0 else "tab:red")
            widths.append(max(0.8, min(4.0, abs(w) * 2.5)))
        nx.draw_networkx_edges(
            graph,
            positions,
            edge_color=edge_colors,
            width=widths,
            arrows=True,
            arrowstyle="->",
            ax=self.ax_network,
            connectionstyle="arc3,rad=0.05",
        )
        labels = {nid: str(nid) for nid in graph.nodes()}
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=9, ax=self.ax_network)
        self.ax_network.set_axis_off()
        if legend_handles:
            self.ax_network.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=9,
                title="Type de nœud",
            )

    def draw_surface_panel(self, genome: Genome, current_inputs: Tuple[float, float]) -> None:
        Z = np.empty((self.grid_n, self.grid_n), dtype=float)
        for iy, y in enumerate(self.grid_y):
            for ix, x in enumerate(self.grid_x):
                Z[iy, ix] = genome.forward([float(x), float(y)])
        self.surface_im.set_data(Z)
        if self.surface_contour is not None:
            collections = getattr(self.surface_contour, "collections", None)
            if collections is not None:
                for coll in collections:
                    coll.remove()
            else:
                remover = getattr(self.surface_contour, "remove", None)
                if callable(remover):
                    remover()
        self.surface_contour = self.ax_surface.contour(
            self.grid_x,
            self.grid_y,
            Z,
            levels=[0.5],
            colors="white",
            linewidths=1.5,
        )
        self.case_marker.set_offsets([current_inputs])

    def on_click(self, event) -> None:
        if event.inaxes != self.ax_ecosystem:
            return
        print("[INFO] Panneaux droits verrouillés sur le meilleur MSE de la génération.")

    def on_key_press(self, event) -> None:
        if event.key == "n":
            print("[INFO] Navigation désactivée : affichage fixé sur le meilleur MSE.")
        elif event.key == "c":
            self.case_index = (self.case_index + 1) % len(XOR_CASES)
            self.update_network_and_surface()
            self.fig.canvas.draw_idle()
        elif event.key == "p":
            self.paused = not self.paused
            state = "en pause" if self.paused else "repris"
            print(f"[INFO] Animation {state}.")
            if self.current_snapshot:
                self.set_ecosystem_title(self.current_snapshot)
                self.fig.canvas.draw_idle()
        elif event.key == "q":
            print("[INFO] Arrêt demandé par l'utilisateur.")
            self.running = False
            plt.close(self.fig)

    def cycle_ranked_member(self) -> None:
        print("[INFO] Sélection manuelle désactivée (meilleur MSE affiché).")

    def capture_frame_if_needed(self) -> None:
        if not self.save_gif_path:
            return
        canvas = self.fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        frame: Optional[np.ndarray] = None
        if hasattr(canvas, "tostring_rgb"):
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            if buf.size == width * height * 3:
                frame = buf.reshape(height, width, 3)
        if frame is None:
            renderer = getattr(canvas, "renderer", None)
            if renderer is None and hasattr(canvas, "get_renderer"):
                renderer = canvas.get_renderer()
            if renderer is not None and hasattr(renderer, "buffer_rgba"):
                rgba = np.asarray(renderer.buffer_rgba(), dtype=np.uint8)
                if rgba.ndim == 3 and rgba.shape[2] >= 3:
                    frame = rgba[..., :3]
                elif rgba.ndim == 1 and rgba.size == width * height * 4:
                    frame = rgba.reshape((height, width, 4))[..., :3]
        if frame is None:
            print("[WARN] Impossible de capturer une frame : backend non supporté.")
            return
        frame = frame.copy()
        self._captured_frames.append(frame)

    def export_gif_if_needed(self) -> None:
        if not self.save_gif_path or not self._captured_frames:
            return
        try:
            import imageio
        except ImportError:
            print(
                "[WARN] imageio non installé : impossible d'enregistrer le GIF "
                f"({self.save_gif_path})."
            )
            return
        wrote = False
        try:
            imageio.mimsave(self.save_gif_path, self._captured_frames, fps=self.gif_fps)
            wrote = True
        except TypeError:
            try:
                duration = 1.0 / max(1, self.gif_fps)
                imageio.mimsave(self.save_gif_path, self._captured_frames, duration=duration)
                wrote = True
            except Exception as exc_inner:
                print(
                    f"[WARN] Échec lors de l'écriture du GIF ({self.save_gif_path}) : "
                    f"{exc_inner}"
                )
                return
        except Exception as exc:
            print(f"[WARN] Échec lors de l'écriture du GIF ({self.save_gif_path}) : {exc}")
            return
        if wrote:
            print(f"[INFO] GIF de l'entraînement sauvegardé → {self.save_gif_path}")

    def set_ecosystem_title(self, snapshot: GenerationSnapshot) -> None:
        status = "pause" if self.paused else "lecture"
        self.ax_ecosystem.set_title(
            f"Génération {snapshot.generation} | best MSE={snapshot.best_mse:.4f} | "
            f"mean MSE={snapshot.mean_mse:.4f} | état={status}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualisation interactive NEAT (3 panneaux) pour la tâche XOR."
    )
    parser.add_argument("--population", type=int, default=300, help="Taille de la population initiale.")
    parser.add_argument(
        "--max-generations", type=int, default=500, help="Nombre maximum de générations."
    )
    parser.add_argument(
        "--success-mse",
        type=float,
        default=0.01,
        help="Seuil MSE pour arrêter l'évolution.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche les logs de génération dans la console.",
    )
    parser.add_argument(
        "--save-gif",
        type=str,
        default=None,
        help="Chemin de sortie (.gif) pour enregistrer l'animation complète.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="Images par seconde pour le GIF (avec --save-gif).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    tracker = InnovationTracker()
    cfg = {
        "population": args.population,
        "elitism_per_species": 1,
        "mutation": {
            "p_mutate_weight": 0.8,
            "p_reset_weight": 0.1,
            "weight_sigma": 0.5,
            "p_add_conn": 0.2,
            "p_add_node": 0.2,
        },
    }
    speciator = Speciator(
        c1=1.0,
        c2=1.0,
        c3=0.4,
        delta_threshold=2.0,
        target_species=6,
        adjust_every=10,
        adjust_step=0.2,
    )

    make_initial_population = make_initial_population_factory(cfg["population"], tracker)
    population = make_initial_population()

    visualizer = NeatLiveVisualizer(
        population,
        tracker,
        speciator,
        cfg,
        max_generations=args.max_generations,
        success_mse=args.success_mse,
        seed=args.seed,
        verbose=args.verbose,
        save_gif_path=args.save_gif,
        gif_fps=args.gif_fps,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
