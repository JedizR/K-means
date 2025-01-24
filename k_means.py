import matplotlib.pyplot as plt
import random
import time

centroids = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(3)]

def generate_points(centroids, num):
    points = []
    for cenx, ceny in centroids:
        for _ in range(num):
            x = random.gauss(cenx, 10)
            y = random.gauss(ceny, 10)
            points.append((x, y))
    return points

points = generate_points(centroids,100)

def plot_points(points, centroids, output_file):
    x_points, y_points = zip(*points)
    x_centroids, y_centroids = zip(*centroids)
    plt.scatter(x_points, y_points, alpha=0.6, label="Generated Points", s=10)
    plt.scatter(x_centroids, y_centroids, color='red', label="Centroids", s=100, marker='x')
    plt.legend()
    plt.title("Generated Points and Centroids")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(output_file)
    plt.show()
    
plot_points(points, centroids, 'points.png')

def k_means_normal(points, k, iterations):
    centroids = [points[random.randint(0, len(points) - 1)] for _ in range(k)]
    
    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
    
        for x,y in points:
            distances = [((x - cx) ** 2 + (y - cy) ** 2) for cx, cy in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append((x, y))
        
        new_centroids = []
        for cluster in clusters:
                if cluster:
                    cluster_x, cluster_y = zip(*cluster)
                    new_centroids.append((sum(cluster_x) / len(cluster_x), sum(cluster_y) / len(cluster_y)))
                else:
                    new_centroids.append(points[random.randint(0, len(points) - 1)])
            
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    return centroids, clusters

start_time = time.time()
final_centroids, final_clusters = k_means_normal(points, k=3, iterations=500)
end_time = time.time()
print("K-Means Normal Time:", end_time - start_time, "seconds")

def plot_clusters(points, clusters, centroids, output_file):
    colors = ["orange", "green", "blue"]

    for i, cluster in enumerate(clusters):
        if cluster:
            cluster_x, cluster_y = zip(*cluster)
            plt.scatter(cluster_x, cluster_y, alpha=0.6, label=f"Cluster {i + 1}", s=10, color=colors[i])
    
    centroid_x, centroid_y = zip(*centroids)
    plt.scatter(centroid_x, centroid_y, color='red', label="Centroids", s=100, marker='x')
    
    plt.legend()
    plt.title("Normal K-Means Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(output_file)
    plt.show()

plot_clusters(points, final_clusters, final_centroids, 'normal_kmeans_clusters.png')

