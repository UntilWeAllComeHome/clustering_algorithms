import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from pointpats import k_test
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score

### Evaluation Functions ###
def plot_ripley_k_pointpats(data_path: str, output_title: str):
    """
    Reads geospatial data, calculates Ripley's K with pointpats, and visualizes it.

    Args:
    data_path: Path to the data file (geojson or shapefile).
    plot_output: Path to save the resulting graphic (needs to be saved as a .png)

    Returns:
    None (plots Ripley's K directly).

    Huge thanks to: https://michaelminn.net/tutorials/python-points/index.html
    """

    # Read sample data
    data = gpd.read_file(data_path)

    # Create Column Stack
    points = np.column_stack((data.geometry.x, data.geometry.y))

    # Calculate Ripley's K
    k = k_test(points, keep_simulations=True)

    # Build visualization
    plt.plot(k.support, k.simulations.T, color='navy', alpha=.01)
    plt.plot(k.support, k.statistic, color='red')
    plt.xlabel('Distance')
    plt.ylabel('K Function')
    plt.title(output_title)

    # Save visualization
    plt.savefig(f"{output_title}.png")


def evaluate_silhouette_coeff(value: float):
  if value > 0.7:
    return "Excellent Clustering"
  
  elif 0.5 < value <= 0.7:
    return "Good Clustering"
  
  elif 0.3 <= value <= 0.5:
    return "Fair Clustering"
  
  else:
    return "Poor Clustering"


def evaluate_calinski_harabasz_index(value: float):
  if value > 2:
    return "Excellent Clustering"
  
  elif 1 < value <= 2:
    return "Good Clustering"
  
  elif 0.5 <= value <= 1:
    return "Fair Clustering"
  
  else:
    return "Poor Clustering"


### Clustering Algorithms ###
def kmeans_cluster(input_file: str, num_clusters: int, output_filename: str):
  """
  Clusters a GeoDataFrame using K-means and saves the output as a geojson file.

  Args:
    input_file: Path to the input shapefile. Requires a str.
    num_clusters: Number of clusters to create.
    output_filename: Path to save the output geojson file. Does not need the extension.
  """

  # Read shapefile with Geopandas
  gdf = gpd.read_file(input_file)

  # Extract point geometry
  points = [(p.x, p.y) for p in gdf.geometry]

  # Apply K-means clustering
  kmeans = KMeans(n_clusters=num_clusters).fit(points)
  cluster_labels = kmeans.labels_

  # Add cluster labels to GeoDataFrame
  gdf["cluster"] = cluster_labels

  # Save output GeoDataFrame with additional cluster label as geojson.
  gdf.to_file(f"output_files/{output_filename}_{num_clusters}.geojson")

  return f"Successfully clustered data into {num_clusters} clusters."


def run_kmedoids_pam_clustering(shapefile, k, output_filename: str, max_iter=100):
  """
  Runs K-medoids PAM clustering on a shapefile.

  Args:
    shapefile: Path to the shapefile.
    k: Number of clusters.
    max_iter: Maximum number of iterations for PAM.

  Returns:
    A geopandas GeoDataFrame with cluster labels added as a new column.
  """
  # Read the shapefile
  df = gpd.read_file(shapefile)

  # Extract coordinates
  points = [(p.x, p.y) for p in df.geometry]
  # points = df.geometry.x.values, df.geometry.y.values

  # Run K-medoids PAM clustering
  km = KMedoids(n_clusters=k, max_iter=max_iter, random_state=0).fit(points)
  cluster_labels = km.labels_

  # Add cluster labels to GeoDataFrame
  df['cluster_label'] = cluster_labels

  # Save output GeoDataFrame with additional cluster label as geojson.
  df.to_file(f"output_files/{output_filename}_{k}.geojson")


def dbscan_cluster(data_path, output_filename, eps=.5, min_samples=5):
  """
  Performs DBSCAN clustering on spatial data.

  Args:
    data_path: Path to the spatial data file (expects a shapefile, but can support any GeoPandas readable file).
    eps: Maximum distance between two points to be considered in the same cluster.
    min_samples: Minimum number of points required to form a cluster.

  Returns:
    A GeoDataFrame with a new 'cluster' column containing cluster labels.
  """

  # Read the spatial data
  data = gpd.read_file(data_path)

  # Extract coordinates
  coordinates = [(p.x, p.y) for p in data.geometry]

  # Perform DBSCAN clustering
  db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(coordinates)

  # Add cluster labels to GeoDataFrame
  data["cluster"] = db.labels_

  # Save file
  data.to_file(f"output_files/{output_filename}_{min_samples}.geojson")


def evaluate_clusters(data_path, n_clusters):
  """
  Reads geospatial data, performs K-means clustering, and calculates cluster evaluation metrics.

  Args:
    data_path: Path to the data file (geojson or shapefile).
    n_clusters: Number of clusters.

  Returns:
    A dictionary containing the Silhouette Coefficient and Calinski-Harabasz Index.
  """

  # Read data
  data = gpd.read_file(data_path)

  # Extract coordinates
  coordinates = [(p.x, p.y) for p in data.geometry]

  # Perform K-means clustering
  kmeans = KMeans(n_clusters=n_clusters).fit(coordinates)
  cluster_labels = kmeans.labels_

  # Calculate evaluation metrics
  silhouette_coeff = silhouette_score(coordinates, cluster_labels)
  calinski_harabasz_index = calinski_harabasz_score(coordinates, cluster_labels)

  # Return results
  return {"silhouette_coefficient": silhouette_coeff,
          "sc_evaluation": evaluate_silhouette_coeff(silhouette_coeff),
          "calinski_harabasz_index": calinski_harabasz_index,
          "chi_evaluation": evaluate_calinski_harabasz_index(calinski_harabasz_index)}


### Example Usage ###
"""
input_data = "input_file/test_points.shp"
output_title = "Input Points Ripley K Results"

plot_ripley_k_pointpats(input_data, output_title)
"""