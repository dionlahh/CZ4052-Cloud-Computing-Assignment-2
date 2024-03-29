import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_directed_graph(adjacency_matrix, graph_name="Graph with Edge Labels"):
    G = nx.DiGraph()
    n = len(adjacency_matrix)
    G.add_nodes_from(range(n))
    for row in range(n):
        for col in range(n):
            if adjacency_matrix[row][col] == 1:
                G.add_edge(col, row)

    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, seed=5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f"arc3, rad = {0.1}")
    plt.title(graph_name)
    plt.show()


def plot_r_history(r_history):
    for j, scores in enumerate(r_history[0]):
        plt.plot([iteration[j] for iteration in r_history], label=f"Node {j}")

    plt.xlabel("Iterations")
    plt.ylabel("PageRank")
    plt.title("PageRank of each Node across Iterations")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()