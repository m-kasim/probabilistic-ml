import numpy as np
import matplotlib.pyplot as plt

class DirichletProcessMixtureModel:
    def __init__(self, alpha, base_distribution, num_data_points):
        self.alpha = alpha  # Concentration parameter
        self.base_distribution = base_distribution  # Base distribution
        self.num_data_points = num_data_points  # Number of data points
        self.data = []  # Data points
        self.cluster_assignments = []  # Cluster assignments
        self.cluster_means = []  # Cluster means

    def generate_data(self):
        """
        Generate data points and their cluster assignments using the Dirichlet Process.
        """
        for i in range(self.num_data_points):
            # Calculate probabilities for existing clusters and a new cluster
            if len(self.cluster_means) > 0:
                cluster_sizes = [self.cluster_assignments.count(k) for k in range(len(self.cluster_means))]
                probabilities = np.array(cluster_sizes + [self.alpha]) / (i + self.alpha)
            else:
                probabilities = [1.0]

            cluster = np.random.choice(len(probabilities), p=probabilities)

            # Create a new cluster if needed
            if cluster == len(self.cluster_means):
                new_mean = self.base_distribution()
                self.cluster_means.append(new_mean)

            # Sample a data point from the chosen cluster
            mean = self.cluster_means[cluster]
            data_point = np.random.normal(mean, 1)
            self.data.append(data_point)
            self.cluster_assignments.append(cluster)

    def plot_results(self):
        """
        Plot the data points with their cluster assignments.
        """
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.cluster_means)))

        for cluster, color in zip(range(len(self.cluster_means)), colors):
            cluster_data = np.array(self.data)[np.array(self.cluster_assignments) == cluster]
            indices = np.where(np.array(self.cluster_assignments) == cluster)[0]
            plt.scatter(
                indices,
                cluster_data,
                label=f"Cluster {cluster + 1}",
                color=color,
                alpha=0.7
            )

        plt.title("Dirichlet Process Mixture Model")
        plt.xlabel("Index")
        plt.ylabel("Data Point Value")
        plt.legend()
        plt.show()


# Parameters
alpha = 1.0  # Concentration parameter
num_data_points = 500  # Number of data points

# Base distribution for cluster means
def base_distribution():
    return np.random.uniform(-10, 10)

# Create and run the corrected DPMM with plot
dpmm_corrected = DirichletProcessMixtureModel(alpha=alpha, base_distribution=base_distribution, num_data_points=num_data_points)
dpmm_corrected.generate_data()
dpmm_corrected.plot_results()

