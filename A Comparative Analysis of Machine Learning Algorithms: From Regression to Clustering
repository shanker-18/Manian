pip install numpy pandas matplotlib seaborn scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Generate synthetic datasets
regression_data = make_regression(n_samples=200, n_features=1, noise=0.1, random_state=0)
classification_data = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, random_state=0)
clustering_data = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)

# Unpack datasets
X_reg, y_reg = regression_data
X_class, y_class = classification_data
X_clust, _ = clustering_data

# Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=0)

# Split classification data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.3, random_state=0)

# Regression Model
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

# Classification Model
clf_model = LogisticRegression()
clf_model.fit(X_train_class, y_train_class)
y_pred_class = clf_model.predict(X_test_class)
clf_accuracy = accuracy_score(y_test_class, y_pred_class)

# Clustering Model
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_clust)

# Visualization
plt.figure(figsize=(18, 12))

# Plot Regression
plt.subplot(3, 3, 1)
plt.scatter(X_test_reg, y_test_reg, color='blue', label='True Values')
plt.scatter(X_test_reg, y_pred_reg, color='red', label='Predictions')
plt.title(f'Regression Model\nMSE: {reg_mse:.2f}')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

# Plot Classification
plt.subplot(3, 3, 2)
sns.scatterplot(x=X_test_class[:, 0], y=X_test_class[:, 1], hue=y_pred_class, palette='coolwarm', s=100, edgecolor='k')
plt.title(f'Classification Model\nAccuracy: {clf_accuracy:.2f}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot Clustering
plt.subplot(3, 3, 3)
plt.scatter(X_clust[:, 0], X_clust[:, 1], c=clusters, cmap='viridis')
plt.title('Clustering Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Add titles and format plots
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, i*3 + j + 1)
        plt.grid(True)
        plt.tight_layout()

plt.show()
