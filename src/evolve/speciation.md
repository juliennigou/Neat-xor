# Spéciation

## Distance δ
- Formule : `δ = (c1 * E) / N + (c2 * D) / N + c3 * W̄`.
- `E` : gènes excessifs ; `D` : gènes disjoints ; `W̄` : différence moyenne de poids (gènes alignés par innovation).
- `N = max(1, nombre de connexions du génome le plus grand)`.

## Species
- **Champs** : `id`, `representative:Genome`, `members:[Genome]`, `age`, `best_fitness`.
- **Méthodes** :
  - `assign(genome, delta_threshold) -> bool` : accepte le génome si `δ ≤ seuil`.
  - `update_representative()` : choisir un nouveau représentant (aléatoire ou meilleur membre).
  - `adjusted_fitnesses() -> list[float]` : partage de fitness intra-espèce.

## Processus de spéciation
- Vider `members` de chaque espèce.
- Parcourir tous les génomes :
  - Tenter l'assignation à la première espèce respectant `δ ≤ delta_threshold`.
  - Créer une nouvelle espèce sinon.
- Ajuster `delta_threshold` si le nombre d'espèces s'écarte de la cible (`±0.1` tous les `adjust_every`).

## Tests mentaux
- Deux génomes différant par deux innovations → calcul manuel de `δ` conforme au chantier précédent.
- Ajustement du seuil si `#espèces` dépasse ou reste sous la cible.
