import pandas as pd
from sklearn.decomposition import PCA
import joblib

# Load the final 1D data
data = pd.read_csv('Final_Dataset/Without_dimensionality_reduction/final_1D.csv', header=None)

# Perform PCA
pca = PCA(n_components=400)
pca.fit(data)

# Save the PCA model
joblib.dump(pca, 'Deployment/PCA_ECG (1).pkl')
print("PCA model saved as 'Deployment/PCA_ECG (1).pkl'")