import numpy as np
from multiprocessing import Pool
from itertools import repeat

d = 0.85
NUM_PROCESSES = 8


def load_graph(file_path):
    """
    Load a directed graph from a file and return the graph structures.
    """
    graph = {}
    reverse_graph = {}
    page_rank = {}
    n_plus = {}

    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                line = line.split('	')
                u = int(line[0])
                v = int(line[1])
                if v not in graph:
                    graph[v] = set()
                    reverse_graph[v] = set()
                if u not in graph:
                    graph[u] = set()
                    reverse_graph[u] = set()
                graph[u].add(v)
                reverse_graph[v].add(u)

    vertices = set(graph.keys()).union(reverse_graph.keys())
    number_of_vertices = len(vertices)

    for vertex in vertices:
        page_rank[vertex] = 1 / number_of_vertices
        n_plus[vertex] = len(graph[vertex]) if vertex in graph else 0

    return graph, reverse_graph, page_rank, n_plus, number_of_vertices


def page_rank_op(current_key, page_rank_local, reverse_graph, n_plus):
    """
    Calculate the sum of contributions from incoming neighbors for the current node.
    """
    sum_contributions = 0
    for v in reverse_graph[current_key]:
        sum_contributions += page_rank_local[v] / (n_plus[v] if n_plus[v] != 0 else 1)
    return sum_contributions


def compute_page_rank_for_node(args):
    """
    Worker function to compute the new PageRank for a single node.
    """
    key, page_rank_local, reverse_graph, n_plus, d, num_vertices = args
    new_value = ((1 - d) / num_vertices) + d * page_rank_op(key, page_rank_local, reverse_graph, n_plus)
    return key, new_value


def page_rank_iterations(graph, reverse_graph, page_rank, n_plus, number_of_vertices):
    """
    Perform the PageRank iterations in parallel until convergence.
    """
    threshold = 1e-3
    pool = Pool(NUM_PROCESSES)  # Create a pool with 8 processes

    vertices = set(graph.keys()).union(reverse_graph.keys())

    while True:
        max_change = 0
        args = [(key, page_rank, reverse_graph, n_plus, d, number_of_vertices) for key in vertices]
        new_page_rank_results = pool.map(compute_page_rank_for_node, args)

        # Update PageRank and calculate the maximum change
        new_page_rank = {}
        for key, new_value in new_page_rank_results:
            max_change = max(max_change, abs(new_value - page_rank[key]))
            new_page_rank[key] = new_value

        page_rank.update(new_page_rank)

        if max_change < threshold:
            break

    pool.close()
    pool.join()
    return page_rank


if __name__ == "__main__":
    file_path = "web-BerkStan.txt"  # Path to the graph file

    print("Preparing graph...")
    graph, reverse_graph, page_rank, n_plus, number_of_vertices = load_graph(file_path)
    print("Graph is ready!")
    for key, value in list(graph.items())[:1]:
        print(f"{key}: {value}")
    for key, value in list(reverse_graph.items())[:1]:
        print(f"{key}: {value}")
    for key, value in list(page_rank.items())[:1]:
        print(f"{key}: {value}")
    for key, value in list(n_plus.items())[:1]:
        print(f"{key}: {value}")
    
    page_rank = page_rank_iterations(graph, reverse_graph, page_rank, n_plus, number_of_vertices)

    print("Final PageRank values:")
    count = 0
    for node, rank in sorted(page_rank.items(), key=lambda item: item[1], reverse=True):
        if count <= 10:
            print(f"Node {node}: {rank}")
            count += 1
