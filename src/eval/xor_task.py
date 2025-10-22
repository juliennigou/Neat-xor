# 1. Le dataset
def xor_dataset():
    """Retourne la liste des 4 cas du XOR."""
    return [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

# 2. Erreur quadratique moyenne
def mse(genome) -> float:
    data = xor_dataset()
    total = 0.0
    for x, y_true in data:
        y_pred = genome.forward(x)
        total += (y_pred - y_true) ** 2
    return total / len(data)

# 3. Fitness
def fitness_from_mse(mse_value: float) -> float:
    return 1.0 / (1.0 + mse_value)

# 4. Accuracy binaire (optionnelle)
def accuracy_threshold(genome, threshold=0.5) -> int:
    data = xor_dataset()
    correct = 0
    for x, y_true in data:
        y_pred = genome.forward(x)
        if (y_pred >= threshold and y_true == 1.0) or (y_pred < threshold and y_true == 0.0):
            correct += 1
    return correct
