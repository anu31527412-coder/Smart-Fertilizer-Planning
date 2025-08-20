import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("../data/soil_dataset.csv")

# Select features
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Moisture']]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Save results
data.to_csv("../results/clustered_soil.csv", index=False)
print("✅ Clustering completed! Results saved to results/clustered_soil.csv")

# Visualization
plt.scatter(data['Nitrogen'], data['Phosphorus'], c=data['Cluster'], cmap='viridis')
plt.xlabel("Nitrogen")
plt.ylabel("Phosphorus")
plt.title("Soil Clusters")
plt.savefig("../results/clusters.png")
plt.show()
