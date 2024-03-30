import numpy as np
import random

def generate_web_graph(n, prob_1=0.5, seed=123):
    # Set the seed for the random number generator
    random.seed(seed)

    # Initialize an empty adjacency list
    adjacency_matrix = [[0] * n for _ in range(n)]

    # Generate random connections between nodes
    num_edges = 0
    for i in range(n):
        for j in range(n):
            if random.random() < prob_1:
                adjacency_matrix[i][j] = 1
                num_edges += 1
            else:
                adjacency_matrix[i][j] = 0

    # Print information about the graph
    print("Generated Web Graph Information:")
    print(f"Number of Nodes: {n}")
    print(f"Number of Edges: {num_edges}")

    return adjacency_matrix


def get_transition_matrix(A):
    arr = np.array(A, dtype=float)
    s = []
    for i in range(0, len(A)):
        s.append(np.sum(arr[:, i]))

    M = arr
    for j in range(0, len(A)):
        M[:, j] = M[:, j] / s[j]

    # print(f"Transition Matrix:\n{M}")
    return M