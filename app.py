import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from pointpats import k_test
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from business_logic import evaluate_clustering_quality


# Create the FastAPI app
app = FastAPI()


### Clustering algorithm endpoints ###
@app.post("kmedoids_pam_clustering")
async def kmedoids_clustering(data_file: UploadFile = File(...), k: int = 5, output_title: str = "kemdoids_pam_clustering", max_iter: int =100):
    """
    Runs K-medoids PAM clustering on a shapefile.
    
    Args:
        data_file: Path to the input spatial file.
        k: Number of clusters.
        output_title: Desired output file name (does not need extension)
        max_iter: Maximum number of iterations for PAM.
    
    Returns:
        Saves a geojson file to the designated location.
    """
    try:
        # Read the shapefile
        df = gpd.read_file(data_file)

        # Extract coordinates
        points = [(p.x, p.y) for p in df.geometry]
        # points = df.geometry.x.values, df.geometry.y.values

        # Run K-medoids PAM clustering
        km = KMedoids(n_clusters=k, max_iter=max_iter, random_state=0).fit(points)
        cluster_labels = km.labels_

        # Add cluster labels to GeoDataFrame
        df['cluster_label'] = cluster_labels

        # Save output GeoDataFrame with additional cluster label as geojson.
        df.to_file(f"output_files/{output_title}_{k}.geojson")

        return {"message": f"K-Medoid PAM cluster saved as {output_title}.geojson"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("kmeans_clustering")
async def kmeans_clustering(data_file: UploadFile = File(...), k: int = 5, output_title: str = "kmeans_clustering"):
    """
    Runs K-Means clustering on an input spatial file.
    
    Args:
        data_file: Path to the input spatial file.
        k: Number of clusters.
        output_title: Desired output file name (does not need extension)
    
    Returns:
        Saves a geojson file to the designated location.
    """
    try:
        # Read the spatial file
        gdf = gpd.read_file(data_file)

        # Extract point geometry
        points = [(p.x, p.y) for p in gdf.geometry]

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k).fit(points)
        cluster_labels = kmeans.labels_

        # Add cluster labels to GeoDataFrame
        gdf["cluster"] = cluster_labels

        # Save output GeoDataFrame with additional cluster label as geojson.
        gdf.to_file(f"output_files/{output_title}_{k}.geojson")

        return {"message": f"K-Means clustering saved as {output_title}.geojson"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("dbscan_clustering")
async def dbscan(data_file: UploadFile = File(...), eps: int = 5, min_samples: int = 5, output_title: str = "dbscan_clustering"):
    """
    Runs DBSCAN clustering on an input spatial file.
    
    Args:
        data_file: Path to the input spatial file.
        eps:
        min_samples:
        output_title: Desired output file name (does not need extension)
    
    Returns:
        Saves a geojson file to the designated location.
    """
    try:
        # Read the spatial data
        data = gpd.read_file(data_file)

        # Extract coordinates
        coordinates = [(p.x, p.y) for p in data.geometry]

        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(coordinates)

        # Add cluster labels to GeoDataFrame
        data["cluster"] = db.labels_

        # Save file
        data.to_file(f"output_files/{output_title}_{min_samples}.geojson")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


### Evaluation Endpoint ###
@app.post("/plot_ripley_k")
async def plot_ripley_k(data_file: UploadFile = File(...), output_title: str = "Ripleys_K"):
    try:
        # Read the data file
        data = gpd.read_file(data_file.file)

        # Extract coordinates
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

        return {"message": f"Ripley's K plot saved as {output_title}.png"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_clusters")
async def evaluate_clusters(data_path: str, clusters: int = 5):
    """
    Reads geospatial data, performs K-means clustering, and calculates cluster evaluation metrics.

    Args:
        data_path: Path to the data file (geojson or shapefile).
        n_clusters: Number of clusters.

    Returns:
        A dictionary containing the Silhouette Coefficient and Calinski-Harabasz Index.
    """
    try:
        # Read data
        data = gpd.read_file(data_path)

        # Extract coordinates
        coordinates = [(p.x, p.y) for p in data.geometry]

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=clusters).fit(coordinates)
        cluster_labels = kmeans.labels_

        # Calculate evaluation metrics
        silhouette_coeff = silhouette_score(coordinates, cluster_labels)
        calinski_harabasz_index = calinski_harabasz_score(coordinates, cluster_labels)

        # Return results
        evaluation_results = {"silhouette_coefficient": silhouette_coeff,
                "sc_evaluation": evaluate_clustering_quality("Silhouette Coefficient", silhouette_coeff),
                "calinski_harabasz_index": calinski_harabasz_index,
                "chi_evaluation": evaluate_clustering_quality("Calinski-Harabasz Index", calinski_harabasz_index)}
        
        return {"message": evaluation_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))