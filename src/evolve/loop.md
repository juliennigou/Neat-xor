# Boucle Évolutionnaire

## Entrées
- Population courante, configuration, tracker d'innovation.

## Sorties
- Nouvelle population.
- Journaux agrégés.
- Checkpoints éventuels.

## Étapes obligatoires
1. Évaluer chaque génome (`mse`, `fitness`, `accuracy_th`).
2. Spécier (assignation + fitness sharing).
3. Appliquer l'élitisme (collecter les élites).
4. Calculer les quotas d'enfants par espèce.
5. Pour chaque espèce :
   - Déterminer si l'enfant naît par mutation seule ou via crossover.
   - Si crossover : sélectionner deux parents (tournoi).
   - Générer l'enfant et appliquer les mutations requises.
6. Assembler la nouvelle population (inclure les élites).
7. Journaliser les métriques et sauvegarder un checkpoint du meilleur si nécessaire.
8. Vérifier les critères d'arrêt (`mse_best ≤ success_mse`, `gen ≥ max_generations`, stagnation `≥ patience`).

## Tests mentaux
- Sur une mini-population (10 individus), dérouler une génération : tailles d'espèces, quotas, nombre d'enfants, élites présents.
