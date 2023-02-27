import numpy as np
from random import sample, shuffle
import matplotlib.pyplot as plt
import pandas as pd


class Kmeans:
    def __init__(self, k, max_iter, dataset, print_info):
        self.k = k
        self.max_iter = max_iter
        self.dataset = dataset
        self.print_info = print_info

        self.centroids = []
        self.clusters = [[] for _ in range(self.k)]

    def _init_centroids(self):
        # use random.shuffle and mix the values in the dataset for better centroids placement
        shuffle(self.dataset)
        # use random.sample to pick k number of centroids
        self.centroids = sample(self.dataset, k=self.k)
        if self.print_info:
            print(f'Centroids = {self.centroids}')

    def _euclidean_distance(self, x_1, y_1, x_2, y_2):
        # use mathematician formula to calculate a distance between two points on a graph
        distance = np.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)
        return distance

    def _assign_to_clusters(self):
        for i, point in enumerate(self.dataset):
            p_x, p_y = point
            distance_to_centroids = []

            for i, centroid in enumerate(self.centroids):
                c_x, c_y = centroid
                distance = self._euclidean_distance(p_x, p_y, c_x, c_y)
                distance_to_centroids.append(distance)

            centroid_idx = np.argmin(distance_to_centroids)
            self.clusters[centroid_idx].append(point)

    def _calculate_mean(self, points):
        sum_x, sum_y = 0, 0

        for i, point in enumerate(points):
            sum_x += point[0]
            sum_y += point[1]

        # Avoid division by zero
        if len(points) > 0:
            new_x, new_y = (np.around(sum_x / len(points)),
                            np.around(sum_y / len(points)))
            new_centroid = [int(new_x), int(new_y)]
            return new_centroid
        else:
            return [0, 0]

    def _recalculate_centroids(self):
        for c_idx, centroid in enumerate(self.centroids):
            self.centroids[c_idx] = self._calculate_mean(self.clusters[c_idx])

    def _plot_data(self, iter_count):
        # Loop through all clusters and plot the data
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_as_df = pd.DataFrame(cluster, columns=['x', 'y'])
            plt.scatter(cluster_as_df['x'], cluster_as_df['y'])
            plt.scatter(self.centroids[cluster_idx][0],
                        self.centroids[cluster_idx][1], s=120, marker="x", c="black")

        plt.title(f'K-means algorithm - Iteration {iter_count}')
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        plt.show()

    def start_clustering(self):
        if self.print_info:
            print(f'Initializing {self.k} random centroids...')

        self._init_centroids()

        previous_centroids = []

        for i in range(self.max_iter):
            if self.print_info:
                print(f'\n--> Iteration {i}')
                print("Assigning the points to the closest clusters...")

            # Empty the clusters before assigning points to them
            self.clusters = [[] for _ in range(self.k)]
            self._assign_to_clusters()

            if self.print_info:
                print("Visualizing data on a plot...")
                self._plot_data(i)

            # Keep track of the previous centroids
            previous_centroids = list(self.centroids)

            if self.print_info:
                print("Recalculating the centroids for better fit...")

            self._recalculate_centroids()

            # Check if the centroids have changed
            if self.centroids == previous_centroids:
                if self.print_info:
                    print("\n=== BEST FIT! ===\n")
                    self._plot_data(f'{i} - Best fit')
                    print(f'--> Centroids = {self.centroids}\n')
                    for cluster_idx, cluster in enumerate(self.clusters):
                        print(f'--> Cluster {cluster_idx + 1} = {cluster}\n')
                break

            if self.print_info:
                print("Visualizing new centroids on a plot...")
                self._plot_data(i)

    def check_accuracy(self):
        if self.print_info:
            print("--> Checking accuracy of clusters...")

        wcss = 0.00
        # calculate the intra-cluster distances
        for c_idx, centroid in enumerate(self.centroids):
            c_x, c_y = centroid

            # Calculate the sum of all the distances between all the points in the cluster
            sum_of_distances = 0.00
            for i, point in enumerate(self.clusters[c_idx]):
                p_x, p_y = point
                distance_to_centroid = self._euclidean_distance(
                    p_x, p_y, c_x, c_y)
                sum_of_distances += distance_to_centroid ** 2

            # Avoid division by zero
            if len(self.clusters[c_idx]) > 0:
                wcss_of_cluster = sum_of_distances / len(self.clusters[c_idx])
                wcss += wcss_of_cluster
            else:
                wcss_of_cluster += 0.00

            if self.print_info:
                print(
                    f'= WCSS of cluster {c_idx+1} = {wcss_of_cluster}')

        # Average WCSS of all clusters
        wcss = wcss / self.k
        print(f'--> Average WCSS across all the clusters = {wcss}')
        return wcss
