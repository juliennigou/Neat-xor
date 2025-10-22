# Genome

## NodeGene
- **Champs** : `id:int`, `type` ∈ {`input`, `bias`, `hidden`, `output`}.
- **Invariants** : IDs uniques dans un génome ; le type d'un nœud est immuable.

## ConnectionGene
- **Champs** : `in:int`, `out:int`, `weight:float`, `enabled:bool`, `innovation:int`.
- **Invariants** :
  - Une seule connexion (`in`, `out`) par génome.
  - `enabled=false` conserve la connexion mais elle ne transmet pas de signal.
  - Les innovations sont des identifiants globaux (cf. `core/innovation.md`).

## Genome
- **Champs** :
  - `nodes:list[NodeGene]`
  - `conns:list[ConnectionGene]`
  - `metadata:dict` (ex : fitness, `species_id`).
- **Méthodes attendues** :
  - `forward(inputs:[float]) -> float` (délégué à `core/network.md`).
  - `copy() -> Genome`.
  - `complexity() -> (n_nodes, n_conns)`.
  - `mutate_weights(cfg)` (cf. `evolve/mutation.md`).
  - `mutate_add_connection(cfg, innovation_tracker)`.
  - `mutate_add_node(cfg, innovation_tracker)`.
  - `to_dict()` / `from_dict()` (pour les checkpoints IO).

## Invariants génome
- Graphe acyclique pour la tâche XOR.
- Toutes les connexions référencent des nœuds existants.
- Aucune connexion `output -> input` (sens interdit dans cette version feed-forward).

## Tests mentaux
- Construire un génome minimal (`A`, `B`, `Bias` → `Y`) et vérifier l'unicité des (`in`, `out`).
- Après `mutate_add_node`, l'ancienne connexion devient `enabled=false`, deux nouvelles connexions apparaissent avec des innovations inédites.
