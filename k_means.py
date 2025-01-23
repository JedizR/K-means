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
print(points)
