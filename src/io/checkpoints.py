# src/io/checkpoints.py
import json, os
from datetime import datetime

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_best(genome, run_dir: str, generation: int, extra: dict | None = None) -> str:
    """
    Sauvegarde un snapshot du meilleur génome.
    Retourne le chemin du fichier écrit.
    """
    _ensure_dir(run_dir)
    payload = {
        "generation": generation,
        "genome": genome.to_dict(),
        "meta": extra or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    path = os.path.join(run_dir, f"best_gen_gen{generation:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def load_genome(path: str, GenomeClass):
    """
    Recharge un génome depuis un checkpoint.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gdict = data["genome"]
    return GenomeClass.from_dict(gdict), data
