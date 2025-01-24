import matplotlib.pyplot as plt
import numpy as np

# use numpy to read csv data then which is a point, plot the scatter
data = np.genfromtxt('/Users/admin/destination/Python-3-2025/K-means/guess the clusters/Points Data.csv', delimiter=',')
plt.scatter(data[:, 0], data[:, 1], s=3)
plt.title('Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

def k_means_numpy(points, k, iterations):
    points = np.array(points)
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    for _ in range(iterations):
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
        clusters = [points[distances[:, i] == np.min(distances, axis=1)] for i in range(k)]
        new_centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else points[np.random.randint(len(points))] for cluster in clusters])
        if np.array_equal(new_centroids, centroids):
            break
        print(new_centroids)
        centroids = new_centroids
    return centroids, clusters

centroids, clusters = k_means_numpy(data, k=4, iterations=500)

def plot_clusters_numpy(points, clusters, centroids, output_file):
    colors = ["orange", "green", "blue","yellow","purple","lime","aqua","black"]
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.6, label=f"Cluster {i + 1}", s=10, color=colors[i])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', label="Centroids", s=100, marker='x')
    plt.legend()
    plt.title("NumPy K-Means Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(output_file)
    plt.show()
    
plot_clusters_numpy(data, clusters, centroids, 'clusters.png')

plt.show()