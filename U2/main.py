import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and preprocess the data a
def load_mnist_csv(filepath):
    data = pd.read_csv(filepath, sep=',')
    if 'label' in data.columns:
        features = data.drop(columns=['label']).values
    else:
        raise KeyError("The 'label' column is not found in the dataset.")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

def compute_similarity_matrix(origin_matrix, size):
    S = np.zeros((size, size))
    distances = euclidean_distances(origin_matrix, origin_matrix)
    S = -distances
    np.fill_diagonal(S, np.median(S))
    return S

if __name__ == "__main__":
    csv_file_path = './mnist_train.csv'
    iterations = 50
    num_processes = 8

    mnist_data = load_mnist_csv(csv_file_path)
    subset_size = 200  # I used 300, but it still takes a lot of time. 200 could be a sweet spot for testing
    mnist_data_subset = mnist_data[:subset_size]
    print("Subset shape:", mnist_data_subset.shape)

    pca = PCA(n_components=50)
    reduced_data = pca.fit_transform(mnist_data_subset)
    S = compute_similarity_matrix(reduced_data, subset_size)

    R = np.zeros_like(S)
    A = np.zeros_like(S)

    def responsibility_worker(args):
        i, S, R, A = args
        R_row = np.zeros_like(R[i])
        for k in range(len(R[i])):
            R_row[k] = S[i][k] - max((A[i][kp] + S[i][kp]) for kp in range(len(R[i])) if k != kp)
        return i, R_row

    def availability_worker(args):
        i, S, R, A = args
        A_row = np.zeros_like(A[i])
        rows, cols = R.shape
        for k in range(cols):
            if i != k:
                A_row[k] = min(0, R[k, k] + sum(max(0, R[ip, k]) for ip in range(rows) if ip != i and ip != k))
            else:
                A_row[k] = sum(max(0, R[ip, k]) for ip in range(rows) if ip != k)
        return i, A_row

    with Pool(num_processes) as pool:
        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}")

            responsibility_args = [(i, S, R, A) for i in range(len(R))]
            responsibility_results = pool.map(responsibility_worker, responsibility_args)
            for i, R_row in responsibility_results:
                R[i] = R_row

            availability_args = [(i, S, R, A) for i in range(len(A))]
            availability_results = pool.map(availability_worker, availability_args)
            for i, A_row in availability_results:
                A[i] = A_row

    C = R + A
    cluster_assignments = np.argmax(C, axis=1)

    print("Final Cluster Assignments:", cluster_assignments)
    print("Number of clusters found:", len(np.unique(cluster_assignments)))
