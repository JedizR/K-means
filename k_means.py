import matplotlib.pyplot as plt
import random

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
