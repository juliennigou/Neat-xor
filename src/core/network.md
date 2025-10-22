# Network Execution

## Fonction `forward(inputs:[float]) -> float`
- Entrée attendue : `[A, B]` avec valeurs 0/1 ; un biais implicite de 1.0 s'ajoute séparément.
- Ordonnancement : topologie `inputs/bias → hidden → output` pour respecter l'acyclicité.
- Calcul : chaque nœud non input somme les contributions pondérées de ses entrées actives.
- Activation : fonction sigmoïde appliquée aux nœuds activables.
- Sortie : `y_hat` dans `(0,1)`.

## Invariants
- Le résultat doit être indépendant de l'ordre de stockage des gènes ; seul le graphe logique compte.
- Les connexions avec `enabled=false` n'influent jamais sur le calcul.

## Tests mentaux
- Réseau direct `3→1` : somme, sigmoïde, comparaison avec le calcul manuel.
- Avec insertion d'un nœud caché : l'ancienne connexion désactivée ne contribue plus au flux.
