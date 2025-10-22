# Crossover

## Règles générales
- Aligner les gènes par numéro d'innovation.
- Gènes appariés : poids choisis au hasard entre les deux parents.
- Gènes disjoints ou excessifs : hérités du parent le plus fit.
- Gènes désactivés : restent désactivés avec probabilité ≈ 0.75.

## Invariants
- L'enfant ne référence que des nœuds existants (union des nœuds parentaux).
- Aucune connexion en double (`in`, `out`).

## Tests mentaux
- Rejouer l'exemple papier et vérifier que l'enfant correspond aux règles ci-dessus.
