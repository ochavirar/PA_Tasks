import numpy as np
from multiprocessing import Pool
from itertools import repeat
from scipy.sparse import csr_matrix

d = 0.85
NUM_PROCESSES = 8

def load_graph(file_path):
    """
    Load a directed graph from a file into a sparse matrix representation.
    """
    edges = []
    nodes = set()

    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                u, v = map(int, line.split('\t'))
                edges.append((u, v))
                nodes.update([u, v])

    nodes = sorted(nodes)
    node_map = {node: idx for idx, node in enumerate(nodes)}
    reverse_node_map = {idx: node for node, idx in node_map.items()}

    row = [node_map[u] for u, v in edges]
    col = [node_map[v] for u, v in edges]
    data = [1] * len(edges)

    n = len(nodes)
    graph = csr_matrix((data, (row, col)), shape=(n, n))

    return graph, node_map, reverse_node_map

def compute_page_rank(graph, d=0.85, threshold=1e-6, max_iter=100):
    """
    Compute the PageRank values for a graph represented as a sparse matrix.
    """
    n = graph.shape[0]
    ranks = np.full(n, 1 / n, dtype=np.float64)
    teleport = (1 - d) / n

    out_degree = graph.sum(axis=1).A1  
    dangling_nodes = (out_degree == 0)  

    for _ in range(max_iter):
        old_ranks = ranks.copy()

        contributions = graph.T.dot(ranks / np.where(out_degree > 0, out_degree, 1))

        dangling_contribution = ranks[dangling_nodes].sum() / n
        contributions += dangling_contribution

        ranks = teleport + d * contributions

        if np.linalg.norm(ranks - old_ranks, ord=1) < threshold:
            break

    return ranks


if __name__ == "__main__":
    file_path = "web-BerkStan.txt"  # Path to the graph file

    print("Preparing graph...")
    graph, node_map, reverse_node_map = load_graph(file_path)
    print("Graph is ready!")
    
    print("Running PageRank...")
    ranks = compute_page_rank(graph)

    print("Final PageRank values:")
    top_nodes = sorted(
        ((reverse_node_map[idx], rank) for idx, rank in enumerate(ranks)),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for node, rank in top_nodes:
        print(f"Node {node}: {rank}")
