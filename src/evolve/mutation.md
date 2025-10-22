# Mutations

## `mutate_weights`
- Perturbation gaussienne (σ donnée) appliquée avec probabilité `p_mutate_weight`.
- Réinitialisation totale d'un poids avec probabilité `p_reset_weight`.

## `mutate_add_connection`
- Tirer deux nœuds compatibles (respect de la direction, pas de doublon).
- Rejeter toute connexion créant un cycle (réseaux feed-forward uniquement).
- Obtenir l'innovation via le tracker global.

## `mutate_add_node`
- Choisir une connexion activée (`A → B`).
- La désactiver.
- Créer un nouveau nœud caché `H`.
- Ajouter `A → H` avec poids `1.0` et `H → B` avec l'ancien poids ; nouvelles innovations obligatoires.

## Tests mentaux
- `mutate_add_connection` ne crée ni doublon ni cycle.
- `mutate_add_node` génère deux innovations et préserve l'acyclicité.
