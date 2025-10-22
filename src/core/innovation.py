
class InnovationTracker: 
    def __init__(self) -> None:

        # Dictionnaire dont les clés sont des tuples (int, int)
        # et les valeurs sont des entiers (innovation IDs)
        self.mapping: dict[tuple[int, int], int] = {}
        self.next_innovation_id: int = 1

    def get_or_create(self, in_node: int, out_node: int) -> int:

        if in_node == out_node:
            raise ValueError("Une connexion ne peut pas boucler sur elle-même.")

        innovation_id = self.mapping.get((in_node, out_node))

        if innovation_id is not None:
            return innovation_id
        
        self.mapping[(in_node, out_node)] = self.next_innovation_id
        self.next_innovation_id += 1
        return self.mapping[(in_node, out_node)]


    def count(self)-> int: 
        return self.next_innovation_id - 1

    def seen_pairs(self)-> set[tuple[int,int]]:
        """Retourne la liste de toutes les paires (in_node, out_node)."""
        return set(self.mapping.keys())

# tracker = InnovationTracker()

# print(tracker.get_or_create(1, 2))  # 1
# print(tracker.get_or_create(1, 2))  # 1 (déjà créé)
# print(tracker.get_or_create(2, 3))  # 2
# print(tracker.seen_pairs())
# print(tracker.count())