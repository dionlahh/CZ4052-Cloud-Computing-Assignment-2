import numpy as np

def pagerank(adj_matrix, d=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank scores using the iterative approach.
    
    Parameters:
        adj_matrix (numpy.ndarray): The adjacency matrix representing the link structure of the graph.
        d (float): Damping factor (probability that a user will continue clicking on links).
        max_iter (int): Maximum number of iterations for the iterative algorithm.
        tol (float): Convergence threshold (stop iteration when PageRank changes less than this value).
        
    Returns:
        numpy.ndarray: PageRank scores for each node.
        int: Number of iterations until convergence.
    """
    n = adj_matrix.shape[0]  # Number of nodes
    teleport_prob = (1 - d) / n  # Teleportation probability
    
    # Initialize PageRank scores
    pagerank_scores = np.ones(n) / n
    num_iter = 0
    
    # Iterative calculation
    for num_iter in range(max_iter):
        prev_pagerank_scores = pagerank_scores.copy()
        for i in range(n):
            # Calculate the sum of PageRank scores of nodes linking to node i
            incoming_score = np.sum(adj_matrix[:, i] * prev_pagerank_scores)
            # Update PageRank score for node i
            pagerank_scores[i] = teleport_prob + d * incoming_score
        # Check for convergence
        if np.linalg.norm(pagerank_scores - prev_pagerank_scores) < tol:
            break
    
    return pagerank_scores, num_iter + 1

def explore_parameter_configurations(adj_matrix, parameter_configs):
    """
    Explore multiple parameter configurations for the PageRank algorithm.
    
    Parameters:
        adj_matrix (numpy.ndarray): The adjacency matrix representing the link structure of the graph.
        parameter_configs (list of dict): List of dictionaries, where each dictionary contains parameter settings.
        
    Returns:
        dict: A dictionary containing the results of running the PageRank algorithm with each parameter configuration.
              The keys are the parameter configurations, and the values are tuples containing PageRank scores
              and the corresponding number of iterations taken.
    """
    results = {}
    for config in parameter_configs:
        pagerank_scores, num_steps = pagerank(adj_matrix, **config)
        print(f"DEBUG: {num_steps}")
        results[str(config)] = (pagerank_scores, num_steps)
    return results
