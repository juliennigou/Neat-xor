# Tâche XOR

## Dataset
- 4 cas : `(0,0→0)`, `(0,1→1)`, `(1,0→1)`, `(1,1→0)`.

## Metrics
- `mse(genome) -> float` : moyenne de `(y_hat - y)^2` sur les 4 cas.
- `fitness(mse) -> float = 1 / (1 + mse)`.
- `accuracy_th(genome, t=0.5) -> int` dans `[0..4]`.

## Tests mentaux
- Pour un réseau trivial : calculer les 4 sorties, la MSE et la fitness, puis comparer aux valeurs théoriques.
