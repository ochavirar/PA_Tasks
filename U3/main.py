import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.sparse import csr_matrix

d = 0.85
NUM_PROCESSES = cpu_count()

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

def compute_contributions(args):
    """
    Compute contributions for a range of rows in the graph.
    """
    graph_chunk, ranks_chunk, out_degree_chunk = args
    contributions = graph_chunk.T.dot(ranks_chunk / np.where(out_degree_chunk > 0, out_degree_chunk, 1))
    return contributions

def compute_page_rank(graph, d=0.85, threshold=1e-6):
    """
    Compute the PageRank values for a graph represented as a sparse matrix.
    Uses multiprocessing to parallelize the computation.
    """
    n = graph.shape[0]
    ranks = np.full(n, 1 / n, dtype=np.float64)
    teleport = (1 - d) / n
    out_degree = graph.sum(axis=1).A1  
    dangling_nodes = (out_degree == 0)
    
    while True:
        old_ranks = ranks.copy()
        chunk_size = n // NUM_PROCESSES
        
        # Prepare arguments for multiprocessing
        args = [
            (graph[i:i + chunk_size], ranks[i:i + chunk_size], out_degree[i:i + chunk_size])
            for i in range(0, n, chunk_size)
        ]
        
        # Compute contributions in parallel
        with Pool(NUM_PROCESSES) as pool:
            contributions_list = pool.map(compute_contributions, args)
        
        # Aggregate contributions
        contributions = sum(contributions_list)
        
        # Handle dangling nodes
        dangling_contribution = ranks[dangling_nodes].sum() / n
        contributions += dangling_contribution
        ranks = teleport + d * contributions
        
        # Check for convergence
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
