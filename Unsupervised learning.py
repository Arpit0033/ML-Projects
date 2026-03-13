import numpy as np
from sklearn.cluster import KMeans

# Data points
X = np.array([
    [1,2],
    [1,4],
    [1,0],
    [10,2],
    [10,4],
    [10,0]
])

# Create model
kmeans = KMeans(n_clusters=2)

# Train model
kmeans.fit(X)

# Print cluster labels
print("Clusters:", kmeans.labels_)