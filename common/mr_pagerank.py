import common.pagerank as pr
import common.plotting_utils as plt
import numpy as np
import findspark


def mapreduce_pagerank(A, n, max_iter=1000, tol=1e-13):
    findspark.init("/opt/homebrew/Cellar/apache-spark/3.5.1/libexec")
    findspark.find()
    import pyspark
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    conf = SparkConf().setAppName("MapReduce").setMaster("local")
    sc = SparkContext("local", conf=conf)
    spark = SparkSession(sc)

    links = []
    for col in range(n):
        entry = (str(col), [])
        for row in range(n):
            if A[row][col] == 1:
                entry[1].append(str(row))
        links.append(entry)

    # Convert links to RDD
    links_rdd = sc.parallelize(links)

    # Create and initialize the ranks
    ranks = links_rdd.map(lambda node: (node[0], 1.0 / n))

    # Initialize previous ranks to None for the first iteration
    prev_ranks = None

    for i in range(max_iter):
        # Join graph info with rank info and propagate to all neighbors rank scores (rank/(number of neighbors)
        # And add up ranks from all incoming edges
        ranks = (
            links_rdd.join(ranks)
            .flatMap(lambda x: [(neighbor, x[1][1] / len(x[1][0])) for neighbor in x[1][0]])
            .reduceByKey(lambda x, y: x + y)
        )
        # Optionally persist or cache intermediate results for better performance
        ranks.persist()

        # Check for convergence
        if prev_ranks is not None:
            # Compute changes in rank values
            rank_changes = (
                ranks.join(prev_ranks).map(lambda x: abs(x[1][0] - x[1][1])).sum()
            )
            # If all rank changes are below threshold, print convergence and break the loop
            if rank_changes < tol:
                print(f"Convergence achieved at iteration {i+1}.")
                break

        # Set current ranks as previous ranks for the next iteration
        prev_ranks = ranks

    # Collect and print final ranks
    final_ranks = ranks.collect()
    spark.stop()

    result = []
    for i in final_ranks:
        result.append(i[1])

    return np.array(result)
