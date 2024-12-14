from typing import List, Tuple
import numpy as np
import itertools as it
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import time

def extract_data_from_file() -> Tuple[int, List, List]:
    dimension = 0
    widths = []

    lines = []
    
    with open('Y-10_t.txt', 'r') as file:
        for line in file:
            lines.append(line.strip())
    
    dimension = int(lines[0])

    distances = [[0]*dimension]*dimension

    widths = list(lines[1].split(' '))
    widths = [int(x) for x in widths]

    lines.pop(0)
    lines.pop(0)

    i = 0
    for line in lines:
        current_line = line.split(' ')
        current_line = [int(x) for x in current_line]
        distances[i] = current_line
        i += 1

    return dimension, widths, distances    

def d(i, j, widths):
    sum_r = 0
    for k in range(i, j, 1):
        sum_r += widths[k]
    distance = ((widths[i] + widths[j])/2) + sum_r    
    return distance

def cost_function(perm, distances, widths):
    permutation = np.array(perm)
    sum = 0
    for i in range(0, len(permutation)):
        for j in range(i+1, len(permutation)):
            distance = d(permutation[i], permutation[j], widths)
            sum += distances[i][j] * distance
    return sum

def srflp_brute_force(dimension: int, widths: list, distances: list):
    permutations = it.permutations(range(dimension))
    cost_min = np.inf
    permutation_min = None
    for perm in permutations:
        cost = cost_function(perm, distances, widths)
        # print(f"Cost for permutation: {cost}")
        if cost < cost_min:
            cost_min = cost
            permutation_min = perm
    print(f"Minimum cost: {cost_min}")
    print(f"Optimal permutation: {permutation_min}")

#####################################
def evaluate_permutation(args):
    perm, distances, widths = args
    return cost_function(perm, distances, widths)

def srflp_brute_force_parallel(dimension: int, widths: list, distances: list):
    permutations = list(it.permutations(range(dimension)))
    cost_min = float('inf')
    permutation_min = None

    with mp.Pool(processes=8) as pool: 
        results = pool.map(evaluate_permutation, [(perm, distances, widths) for perm in permutations])

    for perm, cost in zip(permutations, results):
        if cost < cost_min:
            cost_min = cost
            permutation_min = perm

    print(f"Minimum cost: {cost_min}")
    print(f"Optimal permutation: {permutation_min}")

def bnb_cost_function_partial(perm, distances, widths):
    sum_cost = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            distance = sum(widths[k] for k in range(i + 1, j + 1))
            sum_cost += distances[perm[i]][perm[j]] * distance
    return sum_cost


#################################
def branch_and_bound_parallel(widths, distances, curr_perm, curr_cost, best_cost, used):
    n = len(widths)
    if len(curr_perm) == n:
        return curr_perm, curr_cost if curr_cost < best_cost else (None, best_cost)
    best_permutation = None
    for i in range(n):
        if used[i] == False:
            new_perm = []
            for element in curr_perm:
                new_perm.append(element)
            new_perm.append(i)

            new_used = []
            for use in used:
                new_used.append(use)

            new_used[i] = True

            new_cost = 0
            new_cost += curr_cost
            for j in range(len(curr_perm)):
                for k in range(j + 1, len(new_perm)):
                    distance += widths[new_perm[k]]
                new_cost += distances[curr_perm[j]][i] * distance
            if new_cost >= best_cost:
                continue
            res_perm, res_cost = branch_and_bound_parallel(widths, distances, new_perm, new_cost, min_cost, new_used)
            if res_cost < best_cost:
                best_permutation = res_perm
                min_cost = res_cost
                print(f"New perm:{best_permutation}")
    return best_permutation, best_cost

def worker_task(args):
    widths, distances, perms = args
    process_best_cost = float('inf')
    process_best_perm = None
    for perm in perms:
        current_permutation = list(perm)
        used_permutaions = [False] * len(widths)
        for p in current_permutation:
            used_permutaions[p] = True
        perm_result, perm_cost = branch_and_bound_parallel(widths, distances, current_permutation, 0, process_best_cost, used_permutaions)
        if perm_cost < process_best_cost:
            process_best_cost = perm_cost
            process_best_perm = perm_result
    return process_best_perm, process_best_cost

def srflp_branch_and_bound_parallel(dimension, widths, distances):
    perms = list(it.permutations(range(dimension)))

    with mp.Pool() as pool:
        results = pool.map(worker_task, [(widths, distances, [perm]) for perm in perms])

    min_cost = float('inf')
    optimal_perm = None
    for perm, cost in results:
        if cost < min_cost:
            min_cost = cost
            optimal_perm = perm
    print(f"Minimum cost: {min_cost}")
    print(f"Optimal permutation: {optimal_perm}")

if __name__ == "__main__":
    dimension, widths, distances = extract_data_from_file()
    # print(f"Dimension: {dimension}\nWidths:{widths}\nDistances:{distances}")
    #################
    start_time = time.time()  # Start time
    srflp_brute_force(dimension, widths, distances)
    end_time = time.time()  # End time
    print(f"Execution time of brute force approach was: {end_time - start_time} seconds")
    #################
    start_time = time.time()  # Start time
    srflp_brute_force_parallel(dimension, widths, distances)
    end_time = time.time()  # End time
    print(f"Execution time of brute force parallel approach was: {end_time - start_time} seconds")
    #################
    start_time = time.time()  # Start time
    srflp_branch_and_bound_parallel(dimension, widths, distances)
    end_time = time.time()  # End time
    print(f"Execution time of bnb parallel approach was: {end_time - start_time} seconds")
        
