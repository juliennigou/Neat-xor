from .network import forward_pass

class NodeGene:

    TYPE_OPTIONS = {"input","bias","hidden","output"}

    def __init__(self,id:int,type:str) -> None:

        if id < 1 :
            raise ValueError("les ids doivent etre positifs superieur ou egale à 1")
        if type not in self.TYPE_OPTIONS:
            raise ValueError(
                            f"Le type '{type}' est invalide. Types possibles : {self.TYPE_OPTIONS}"
                        )
        
        self.id : int = id
        self.type : str = type

    def __repr__(self) -> str:
        """Retourne une représentation lisible et non ambiguë de l’objet."""
        return f"NodeGene(id={self.id}, type='{self.type}')"
    

class ConnectionGene: 
    def __init__(self, in_node: int, out_node: int, weight: float, enabled: bool, innovation: int) -> None:
        if in_node == out_node:
            raise ValueError("Une connexion ne peut pas boucler sur elle-même.")

        if in_node < 1 or out_node < 1:
            raise ValueError("in_node et out_node doivent être ≥ 1.")
        if innovation < 1:
            raise ValueError("innovation doit être ≥ 1.")
        if not isinstance(enabled, bool):
            raise TypeError("enabled doit être un booléen.")

        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation


class Genome:
    def __init__(self):
        # Ensemble des nœuds et connexions
        self.nodes: dict[int, NodeGene] = {}
        self.conns: dict[tuple[int,int], ConnectionGene] = {}

    def add_node(self, node: NodeGene) -> None:
        """Ajoute un nœud au génome si son id n'existe pas déjà."""
        if node.id in self.nodes:
            raise ValueError(f"Le nœud avec id={node.id} existe déjà.")
        
        self.nodes[node.id] = node


    def add_connection(self, conn: ConnectionGene) -> None:
            """Ajoute une connexion entre deux nœuds existants, si elle n’existe pas déjà."""
            key = (conn.in_node, conn.out_node)

            # 1️⃣ Vérifier si la connexion existe déjà
            if key in self.conns:
                raise ValueError(f"La connexion {key} existe déjà.")

            # 2️⃣ Vérifier que les deux nœuds existent
            if conn.in_node not in self.nodes:
                raise ValueError(f"Le nœud d’entrée {conn.in_node} n’existe pas dans le génome.")
            if conn.out_node not in self.nodes:
                raise ValueError(f"Le nœud de sortie {conn.out_node} n’existe pas dans le génome.")

            # 3️⃣ Tout est OK → ajout
            self.conns[key] = conn        

    def complexity(self) -> tuple[int, int]:
            """Retourne le nombre de nœuds et de connexions (nb_nodes, nb_conns)."""
            return len(self.nodes), len(self.conns)


    def forward(self, inputs: list[float]) -> float:
        """Délègue l'exécution à network.forward_pass."""
        return forward_pass(self, inputs)

    def copy(self) -> "Genome":
        """Copie profonde du génome (nodes + conns)."""
        from .genome import NodeGene, ConnectionGene
        g = type(self)()
        # copier nodes
        for nid, n in self.nodes.items():
            g.nodes[nid] = NodeGene(n.id, n.type)
        # copier conns
        for (u, v), c in self.conns.items():
            g.conns[(u, v)] = ConnectionGene(
                in_node=c.in_node,
                out_node=c.out_node,
                weight=float(c.weight),
                enabled=bool(c.enabled),
                innovation=int(c.innovation),
            )
        return g

    def to_dict(self) -> dict:
        """Sérialisation simple (JSON-friendly)."""
        return {
            "nodes": [{"id": n.id, "type": n.type} for n in self.nodes.values()],
            "conns": [{
                "in": c.in_node, "out": c.out_node, "weight": c.weight,
                "enabled": c.enabled, "innovation": c.innovation
            } for c in self.conns.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        from .genome import NodeGene, ConnectionGene
        g = cls()
        # nodes d'abord
        for nd in data.get("nodes", []):
            g.add_node(NodeGene(int(nd["id"]), str(nd["type"])))
        # puis conns
        for cd in data.get("conns", []):
            g.add_connection(ConnectionGene(
                in_node=int(cd["in"]), out_node=int(cd["out"]),
                weight=float(cd["weight"]), enabled=bool(cd["enabled"]),
                innovation=int(cd["innovation"])
            ))
        return g

    def __repr__(self) -> str:
            """Retourne une représentation claire du génome."""
            nodes_str = "\n".join(
                [f"  {node.id}: {node.type}" for node in self.nodes.values()]
            )
            conns_str = "\n".join(
                [
                    f"  ({c.in_node} → {c.out_node}) | w={c.weight:+.3f} | "
                    f"enabled={c.enabled} | innov={c.innovation}"
                    for c in self.conns.values()
                ]
            )
            nb_nodes, nb_conns = self.complexity()
            return (
                f"🧬 Genome: {nb_nodes} nœuds, {nb_conns} connexions\n"
                f"Nodes:\n{nodes_str or '  (aucun)'}\n"
                f"Connexions:\n{conns_str or '  (aucune)'}"
            )



# g = Genome()
# g.add_node(NodeGene(1,"input"))
# g.add_node(NodeGene(2,"input"))
# g.add_node(NodeGene(3,"bias"))
# g.add_node(NodeGene(4,"output"))

# g.add_connection(ConnectionGene(1,4,0.75,True,1))
# g.add_connection(ConnectionGene(2,4,-0.10,True,2))
# g.add_connection(ConnectionGene(3,4,0.50,True,3))

# print(g.complexity())   # (4,3)
