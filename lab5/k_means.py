import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    selected = np.random.choice(data.shape[0], k, replace=False)
    return data[selected, :]

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = np.zeros((k, data.shape[1]))
    centroids[0, :] = data[np.random.choice(data.shape[0], 1), :]
    for i in range(1, k):
        distances = np.sqrt(np.sum((data[:, np.newaxis, :] - centroids[:i, :])**2, axis=2))
        sum_distances = np.sum(distances, axis=1)
        max_index = np.argmax(sum_distances)
        centroids[i, :] = data[max_index, :]
    return centroids

def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    distances = np.sum((data[:, np.newaxis, :] - centroid)**2, axis=2)
    assigned = np.argmin(distances, axis=1)
    return assigned

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    centroids = np.zeros((len(np.unique(assignments)), data.shape[1]))
    for i in np.unique(assignments):
        centroids[i, :] = np.mean(data[assignments==i], axis=0)
    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

