import numpy as np
import random

def simplified_pagerank(M, n, max_iter=1000, tol=1e-13):
    # Prone to spider traps
    r_history = []
    R_new = R_old = np.ones((n, 1)) / n

    for i in range(max_iter):
        R_new = np.matmul(M, R_old)
        # Check for convergence
        diff = np.sum(abs(R_new - R_old))
        if diff < tol:
            print(f"Converged after {i+1} iterations.")
            break
        r_history.append(R_old[:, 0].round(4).tolist())
        R_old = R_new

    print("Final result:\n", R_new[:, 0])
    return R_new[:, 0], r_history


def modified_pagerank(
    M, n, boredom_distribution=None, beta=0.85, max_iter=1000, tol=1e-13
):
    r_history = []
    R_new = (1.0 + np.zeros([len(M), 1])) / len(M)
    if boredom_distribution == None:
        boredom_distribution = (1.0 + np.zeros([len(M), 1])) / len(M)
    print(f"Initial Rank Vector:\n{R_new}")

    c = (1 - beta) * boredom_distribution
    R_old = R_new

    for i in range(max_iter):
        R_new = beta * np.matmul(M, R_old) + c
        # Check for convergence
        diff = np.sum(abs(R_new - R_old))
        if diff < tol:
            print(f"Converged after {i+1} iterations.")
            break
        # r_history.append(R_old.round(4).tolist())
        r_history.append(R_old[:, 0].round(4).tolist())
        R_old = R_new

    print("Final result:\n", R_new[:, 0])
    return R_new[:, 0], r_history


def remove_dead_ends(M):
    n = len(M)
    replaced = False
    replacement = np.ones((n, 1)) / n
    # print(f"replacement:\n{replacement}")

    for j in range(n):  # Iterate over each column
        if np.all(np.isnan(M[:, j])):  # Check if all elements in the column are NaN
            # Replace the NaN elements in the column with the corresponding elements from the replacement column
            M[:, j] = replacement[:, 0]
            replaced = True

    if replaced:
        print("Dead Ends found!")
        print(f"New Transition Matrix:\n{M}")
    else:
        print("No Dead Ends")
    return M


def dead_end_pagerank(M, n, beta=0.85, max_iter=1000, tol=1e-13):
    M = remove_dead_ends(M)
    r_history = []
    R_new = (1.0 + np.zeros([len(M), 1])) / len(M)
    print(f"Initial Rank Vector:\n{R_new}")

    c = (1 - beta) * R_new
    R_old = R_new

    for i in range(max_iter):
        R_new = beta * np.matmul(M, R_old) + c
        # Check for convergence
        diff = np.sum(abs(R_new - R_old))
        if diff < tol:
            print(f"Converged after {i+1} iterations.")
            break
        # r_history.append(R_old.round(4).tolist())
        r_history.append(R_old[:, 0].round(4).tolist())
        R_old = R_new

    print("Final result:\n", R_new[:, 0])
    return R_new[:, 0], r_history


def pagerank(M, n, boredom_distribution=None, beta=0.85, max_iter=1000, tol=1e-13):
    M = remove_dead_ends(M)
    r_history = []
    R_new = (1.0 + np.zeros([len(M), 1])) / len(M)
    if boredom_distribution == None:
        boredom_distribution = (1.0 + np.zeros([len(M), 1])) / len(M)
    # print(f"Initial Rank Vector:\n{R_new}")

    c = (1 - beta) * boredom_distribution
    R_old = R_new

    for i in range(max_iter):
        R_new = beta * np.matmul(M, R_old) + c
        # Check for convergence
        diff = np.sum(abs(R_new - R_old))
        if diff < tol:
            print(f"Converged after {i+1} iterations.")
            break
        # r_history.append(R_old.round(4).tolist())
        r_history.append(R_old[:, 0].round(4).tolist())
        R_old = R_new

    # print("Final result:\n", R_new[:, 0])
    print(f"Sum of PageRank: {sum(R_new[:, 0])}")
    return R_new[:, 0], r_history


def closed_form_pagerank(M, n, boredom_distribution=None, beta=0.85):
    # M = transition matrix
    # b = (1-beta) * boredom distribution
    # R' = (I - beta * M)^-1 * b <== closed form solution
    # Complexity of (I-c1M)^-1 is O(T^3)
    M = remove_dead_ends(M)
    I = np.eye(n)
    if boredom_distribution == None:
        boredom_distribution = (1.0 + np.zeros([len(M), 1])) / len(M)
    c = (1 - beta) * boredom_distribution
    A = np.matmul(np.linalg.inv(I - beta * M), c)

    return A[:, 0]

def iter_vs_closed(r_iter, r_closed):
    if np.allclose(r_iter, r_closed):
        print("Arrays are almost equal.")
        return True, 0.0
    else:
        diff = np.abs(r_iter - r_closed)
        max_error = np.max(diff)
        print("Arrays are not almost equal. Maximum absolute error:", max_error)
        return False, max_error
