# Innovation Tracker

## Rôle
Garantir des IDs d'innovation stables par paire (`in`, `out`) sur l'ensemble de la population.

## API attendue
- `get_or_create(in:int, out:int) -> int`
- `seen_pairs() -> Iterable[(in, out, innovation)]` (debug / introspection).

## Invariant
La même paire (`in`, `out`) reçoit toujours le même numéro durant tout le run.

## Tests mentaux
- Deux génomes distincts demandent `(1, 4)` à des moments différents → même innovation.
- Nouvelle paire `(1, 5)` → innovation inédit.
