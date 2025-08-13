# kmeans_training.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])

# 2. Select features (Annual Income & Spending Score for easy visualization)
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Find optimal k using Elbow Method (optional)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Uncomment to see elbow plot
# import matplotlib.pyplot as plt
# plt.plot(range(1, 11), wcss)
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.title('Elbow Method')
# plt.show()

# 4. Train model with chosen k (example: 5 clusters)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X)

# 5. Save model
with open("model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

descriptions = []
for i, row in centroids.iterrows():
    gender_desc = "mostly male" if row["Gender"] > 0.5 else "mostly female"
    
    if row["Age"] < 30:
        age_desc = "young"
    elif row["Age"] < 50:
        age_desc = "middle-aged"
    else:
        age_desc = "older"
    
    if row["Annual Income (k$)"] < 40:
        income_desc = "low income"
    elif row["Annual Income (k$)"] < 70:
        income_desc = "medium income"
    else:
        income_desc = "high income"
    
    if row["Spending Score (1-100)"] < 40:
        spending_desc = "low spending"
    elif row["Spending Score (1-100)"] < 70:
        spending_desc = "moderate spending"
    else:
        spending_desc = "high spending"
    
    descriptions.append(f"{gender_desc}, {age_desc}, {income_desc}, {spending_desc}")

# Show results
for idx, desc in enumerate(descriptions):
    print(f"Cluster {idx}: {desc}")

# Save descriptions for later use
with open("cluster_descriptions.pkl", "wb") as f:
    pickle.dump(descriptions, f)