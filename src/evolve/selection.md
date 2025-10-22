# Sélection

## Élitisme
- Conserver `elitism_per_species` meilleurs membres de chaque espèce sans modification.

## Quotas d'enfants
- `F_k` : somme des fitness ajustées de l'espèce `k`.
- `offspring_k = round(POP_SIZE * F_k / Σ F_all)`.
- Ajuster les arrondis pour que la somme finale corresponde exactement à `POP_SIZE`.

## Sélection de parents
- Tournoi de taille 3 : tirer 3 membres au hasard et garder le plus fit.
- Répéter le processus pour chaque parent nécessaire au crossover.

## Tests mentaux
- Une espèce portant 60 % de la fitness ajustée globale obtient ≈ 60 % des enfants.
- Le tournoi favorise statistiquement les individus les plus fit.
