# Repository Guidelines

## Project Structure & Module Organization
- `src/core/` contient l’ADN NEAT (genome, innovation tracker, forward pass). Ajoute toute nouvelle logique de réseau à proximité des fichiers existants pour préserver l’API.
- `src/evolve/` héberge la boucle d’entraînement, la spéciation et les opérateurs génétiques; garde chaque responsabilité dans son module (`speciation.py`, `mutation.py`, etc.).
- `src/eval/` rassemble la définition de la tâche XOR et les métriques; toute variante de tâche ou nouvelle métrique va ici.
- `src/io/` gère journalisation, checkpoints et visualisations. `runs/` stocke les sorties générées (exemples légers seulement dans le dépôt).
- Les scripts racine (`train_xor.py`, `run_demo.py`, `viz_*.py`) servent d’entrées CLI; `config/neat.yaml` centralise les hyperparamètres que les scripts consomment.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` prépare un environnement local (Python 3.11 recommandé).
- `python run_demo.py` valide bout à bout innovation, mutations, crossover et spéciation; exécute-le comme test de fumée.
- `python train_xor.py` lance une session NEAT complète et alimente `runs/` et `checkpoints/`; ajoute des options CLI au besoin mais garde la configuration dans YAML.
- `python viz_decision_boundary.py --run runs/<nom>` et `python viz_animate_training.py` produisent les diagnostics visuels issus d’un run.
- `python viz_network_activations.py --genome checkpoints/<fichier>.json` inspecte un génome sauvegardé.

## Coding Style & Naming Conventions
- Respecte PEP 8 : indentation 4 espaces, modules/fonctions en `snake_case`, classes en `PascalCase`, constantes en `UPPER_SNAKE_CASE`, lignes ≤ 100 caractères.
- Étends les annotations de types existantes (`dict[int, NodeGene]`, etc.) et préfère des structures explicites plutôt que des `dict` anonymes.
- Messages d’erreur et docstrings sont courts, informatifs, souvent bilingues; garde l’API publique et les signatures en anglais.
- Utilise des graines (`random.seed`) et les seuils de `config/neat.yaml` pour garder des expériences reproductibles.

## Testing Guidelines
- Aucune suite automatisée n’est livrée; ajoute des tests `pytest` sous `tests/` pour toute logique nouvelle ou modifiée (mutations, sélection, métriques).
- Complète les tests par `python run_demo.py` pour détecter les régressions d’orchestration et vérifier la compatibilité des opérateurs.
- Pour les changements proches de la boucle d’entraînement, exécute `python train_xor.py --max-generations <petit>` avec une graine fixe.

## Commit & Pull Request Guidelines
- Utilise des messages impératifs et ciblés (`evolve: adjust species threshold`, `fix: guard disabled edges`). Mentionne le module si cela clarifie l’impact.
- Dans chaque PR, fournis objectif, lien issue, tests exécutés et, si pertinent, captures ou métriques provenant de `runs/`.
- Nettoie les artefacts volumineux (`runs/`, `checkpoints/`) avant de pousser et veille à aligner `config/neat.yaml` sur les scripts.
