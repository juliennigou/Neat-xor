# NEAT XOR

Implementation-ready scaffolding for NeuroEvolution of Augmenting Topologies (NEAT) applied to the classic XOR benchmark. The repository provides a clean separation between core genetics, evolutionary operators, IO tooling, and several visualization entry points—including a live 3-panel dashboard to watch species and network structure evolve generation after generation.

> **Heads up:** a training GIF will live here once recorded.  
> `![Live training demo](docs/assets/neat-live-training.gif)`

## Highlights
- **Typed NEAT primitives** — genomes, innovation tracking, feed-forward execution, and serialization live under `src/core`.
- **Plug-and-play evolution loop** — evaluation, speciation, selection, mutation, and checkpointing are split into dedicated modules in `src/evolve`.
- **Interactive visuals** — run `viz_live_training.py` for the ecosystem/network/decision-surface dashboard, plus utilities to inspect checkpoints, decision boundaries, and activations.
- **Config-driven experiments** — tweak population size, mutation probabilities, and speciation thresholds from `config/neat.yaml`.
- **Contributor guide** — see `AGENTS.md` for coding standards, testing expectations, and PR etiquette.

## Repository Layout
```
├── config/              # YAML configs (default hyperparameters, seeds)
├── docs/                # Project notes and future assets (GIF placeholder)
├── src/
│   ├── core/            # Genome, connection, network execution, innovation tracker
│   ├── evolve/          # Speciation, selection, mutation, crossover, training loop
│   ├── eval/            # XOR dataset, metrics, accuracy utilities
│   └── io/              # Logging, checkpoints, visualization helpers
├── runs/                # Auto-generated experiment outputs (metrics + checkpoints)
├── train_xor.py         # End-to-end NEAT training pipeline for XOR
├── run_demo.py          # Guided smoke test of every primitive (mutations, crossover, speciation)
├── viz_live_training.py # 3-panel interactive dashboard (species + network + decision surface)
├── viz_decision_boundary.py
├── viz_animate_training.py
└── viz_network_activations.py
```

## Quick Start
1. **Requirements**  
   Python ≥ 3.11 with `numpy`, `matplotlib`, and `networkx`.

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install numpy matplotlib networkx
   ```

3. **Run the integration demo**  
   ```bash
   python run_demo.py
   ```
   Exercises innovation tracking, genome forward pass, mutations, crossover, and speciation in a single script.

## Training the XOR Task
```bash
python train_xor.py
```
- Uses the defaults from `config/neat.yaml` (seed 123, population 150, realistic mutation rates).
- Logs metrics per generation (`runs/<timestamp>/metrics.csv`) and checkpoints the best genome as soon as it improves.
- Training stops early when the best mean-squared error (MSE) drops below the configured success threshold (`xor.success_mse`).

### Customizing Experiments
Edit `config/neat.yaml` before launching a run:
- **Population & elitism**: `population`, `elitism_per_species`
- **Mutation probabilities**: `mutation.p_mutate_weight`, `p_add_conn`, `p_add_node`, etc.
- **Speciation sensitivity**: `speciation.delta_threshold`, `target_species`, `adjust_every`, `adjust_step`
- **Stopping conditions**: `limits.max_generations`, `limits.stagnation_patience`, `xor.success_mse`

You can also invoke scripts with runtime options:
```bash
python viz_live_training.py --population 200 --max-generations 250 --seed 123 --verbose
```

## Visualizing NEAT in Action
- `viz_live_training.py` — live dashboard with:
  - Left panel: species ecosystem (color = MSE, marker = species)
  - Upper-right: selected genome rendered with NetworkX (node shape/color by type & activation)
  - Lower-right: decision surface heatmap with XOR samples highlighted
  - Add `--save-gif docs/assets/neat-live-training.gif` to export the full run as an animated GIF (requires `imageio`; tweak cadence via `--gif-fps`).
- `viz_decision_boundary.py` — render the surface of a saved checkpoint.
- `viz_animate_training.py` — create an animation/GIF from best-genome checkpoints across a run.
- `viz_network_activations.py` — inspect node activations for a stored genome on specific inputs.

Once the training GIF is produced, attach it under `docs/assets` and update the placeholder link above.

## Development Workflow
- Read `AGENTS.md` for style conventions (PEP 8), naming rules, and testing guidance.
- Add `pytest` suites under `tests/` for new logic; the quick smoke test remains `python run_demo.py`.
- When contributing:
  - Use imperative commit messages prefixed by scope when helpful (`core:`, `evolve:`, `viz:`).
  - Summarize intent, link issues, and note tests/screenshots in pull requests.
  - Avoid committing bulky artefacts from `runs/` or `checkpoints/`; include lightweight samples only.

## Roadmap
- Record and embed the live-training GIF in the README.
- Extend `viz_live_training.py` with optional logging overlays (species counts, threshold trends).
- Add regression tests for speciation edge cases and mutation operators.
- Explore adding recurrent links and additional benchmark tasks beyond XOR.

---
Questions, ideas, or findings? Open an issue or drop a note in your pull request—contributions that improve clarity, robustness, or visualization polish are especially welcome. Happy evolving! 🚀
