# Checkpoints

## Contenu requis
- Meilleur génome : nœuds, connexions, poids, flags `enabled`, innovations.
- Configuration utilisée.
- Seed initiale.
- Date et génération.

## Nommage
- `runs/run_YYYYMMDD_HHMM/best_genome_gen_XX.json`.

## Invariants
- Chaque checkpoint doit être auto-suffisant pour restaurer un run (forward immédiat possible).

## Test mental
- Recharger un snapshot et réévaluer → métriques identiques.
