
def evaluate_clustering_quality(metric: str, value: float) -> str:
    """
    Evaluates the quality of a clustering based on a given metric and its value.

    Args:
        metric: The clustering metric (e.g., "Silhouette Coefficient", "Calinski-Harabasz Index").
        value: The calculated value of the metric.

    Returns:
        A string describing the clustering quality based on the metric and value.
    """

    thresholds = {
        "Silhouette Coefficient": {
            0.7: "Excellent",
            0.5: "Good",
            0.3: "Fair",
            0: "Poor",
        },
        "Calinski-Harabasz Index": {
            2: "Excellent",
            1: "Good",
            0.5: "Fair",
            0: "Poor",
        },
    }

    if metric not in thresholds:
        raise ValueError(f"Invalid metric: {metric}")

    for threshold, label in thresholds[metric].items():
        if value >= threshold:
            return f"{label} Clustering ({metric}: {value:.2f})"

    return f"Poor Clustering ({metric}: {value:.2f})"